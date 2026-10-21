// vlx-cli: VLX-Seek-1.5 full chain in C++ (ggml vision graphs + llama.cpp LLM).
//
// Commands:
//   run       — the C++ twin of `python inference.py`: same arguments, same
//               task templates, same prompt/token ids, same greedy decode and
//               the same printed JSON (answer, result_bbox_list, prompt_tokens,
//               completion_tokens, visualization_path). WeDetect-Uni proposals
//               are produced in C++ when --bbox-list is omitted.
//   detect    — WeDetect-Uni proposals only (boxes.json; --dump-dir gate taps).
//   gen       — plain-text generation smoke test.
//   gen-embd  — embedding-injection smoke test (f32 rows file).
//   visualize — draw a boxes.json on an image (inference.py rectangle style).
//
// Precision contract of `run` (every rule mirrors what the official python
// pipeline computes, see vlx_seek_worker.py / modeling_vlx_seek_1_5.py):
//   * host preprocessing is bit-exact (image_prep.hpp), including the bf16
//     casts the model applies to its pixel inputs;
//   * inside a network, ggml computes in fp32 (f16/q8_0 weights where the
//     GGUF says so); where python crosses a module boundary with an explicit
//     .to(bf16) — HFRE -> projector input, vision -> LLM inputs_embeds — the
//     same rounding is applied here;
//   * the LLM sees one prefill batch with M-RoPE positions identical to the
//     official position_ids (all sections = arange), then greedy argmax.
//
// No ggml/llama.cpp source edits here: this file links the public API only;
// source changes live in third_party/{ggml,llama}-patches/*.patch.
#include "ggml_util.hpp"

#include "gguf_loader.hpp"
#include "hfre.hpp"
#include "image_prep.hpp"
#include "mmproj_graph.hpp"
#include "radio_graph.hpp"
#include "simplefpn_graph.hpp"
#include "vlx_types.hpp"
#include "wedetect_graph.hpp"

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// stb_image's implementation lives in image_prep.cpp (single definition).
#include "stb_image.h"
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int default_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n ? int(n) : 4;
}

std::string lower_ascii(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// --device cpu|cuda|vulkan -> the first ggml device whose backend registry
// name matches (case-insensitive).
ggml_backend_dev_t find_device(const std::string& device) {
    const std::string want = lower_ascii(device.empty() ? "cpu" : device);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const std::string reg = lower_ascii(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
        if (reg == want) return dev;
    }
    return nullptr;
}

// Backend handle for --device; CPU gets an explicit thread count (the ggml
// default is 4, which under-uses the host).
ggml_backend_t init_backend(const std::string& device, int n_threads) {
    ggml_backend_dev_t dev = find_device(device);
    if (!dev) {
        std::fprintf(stderr, "no ggml device for --device %s\n", device.c_str());
        return nullptr;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be && lower_ascii(device.empty() ? "cpu" : device) == "cpu")
        ggml_backend_cpu_set_n_threads(be, n_threads);
    if (be)
        std::fprintf(stderr, "backend: %s (%s)\n", ggml_backend_name(be),
                     ggml_backend_dev_description(dev));
    return be;
}

// Device memory in use (total - free) via the backend registry. The counters
// include other processes on the device, so diff a before/after sample to
// attribute memory to this run (see --warm-runs reporting).
size_t device_mem_used(const std::string& device) {
    ggml_backend_dev_t dev = find_device(device);
    if (!dev) return 0;
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(dev, &free_b, &total_b);
    return total_b - free_b;
}

bool read_file(const std::string& path, std::vector<uint8_t>* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0);
    out->resize(size_t(n));
    in.read(reinterpret_cast<char*>(out->data()), n);
    return true;
}

// Creates the parent directory; a failed write is reported, never silent.
bool write_file(const std::string& path, const void* data, size_t bytes) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    std::error_code ec;
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    std::ofstream out(path, std::ios::binary);
    if (out) out.write(reinterpret_cast<const char*>(data), std::streamsize(bytes));
    if (!out) {
        std::fprintf(stderr, "write failed: %s\n", path.c_str());
        return false;
    }
    return true;
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// ---------------- python-compatible JSON in/out ----------------

// json.loads of a list of numeric lists ([[x1, y1, x2, y2], ...]) or a flat
// numeric list ([2, 3]). depth = 2 / 1. Returns false on malformed input.
bool parse_json_numbers(const std::string& s, int depth, std::vector<std::vector<double>>* out) {
    size_t i = 0;
    auto ws = [&]() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; };
    auto expect = [&](char c) { ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; };
    auto number_list = [&](std::vector<double>* v) {
        if (!expect('[')) return false;
        if (expect(']')) return true;
        do {
            ws();
            char* end = nullptr;
            const double d = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) return false;
            i = size_t(end - s.c_str());
            v->push_back(d);
        } while (expect(','));
        return expect(']');
    };
    out->clear();
    if (depth == 1) {
        out->emplace_back();
        if (!number_list(&out->back())) return false;
    } else {
        if (!expect('[')) return false;
        if (!expect(']')) {
            do {
                out->emplace_back();
                if (!number_list(&out->back())) return false;
            } while (expect(','));
            if (!expect(']')) return false;
        }
    }
    ws();
    return i == s.size();
}

// float.__repr__: shortest round-trip digits, fixed notation in
// [1e-4, 1e16) with a mandatory ".0", else 'e' notation ("1e-05").
std::string py_float(double v) {
    char buf[64];
    const double a = std::fabs(v);
    if (a == 0.0 || (a >= 1e-4 && a < 1e16)) {
        const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::fixed);
        std::string s(buf, r.ptr);
        if (s.find('.') == std::string::npos) s += ".0";
        return s;
    }
    const auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    return std::string(buf, r.ptr);
}

// json.dumps(..., ensure_ascii=False) string escaping.
std::string py_json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += char(c);
                }
        }
    }
    return o + "\"";
}

// Official-format boxes.json: [[x1, y1, x2, y2], ...] as json.dump(tensor.tolist()).
std::string boxes_to_json(const std::vector<vlx::Box>& boxes) {
    std::string s = "[";
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& b = boxes[i];
        s += (i ? ", [" : "[") + py_float(b.x1) + ", " + py_float(b.y1) + ", " +
             py_float(b.x2) + ", " + py_float(b.y2) + "]";
    }
    return s + "]";
}

// ---------------- worker semantics (vlx_seek_worker.py) ----------------

using BoxD = std::array<double, 4>;   // python float xyxy (caller-facing)

// _validate_boxes: clip to the image, sort each axis, reject empty boxes.
bool validate_boxes(const std::vector<std::vector<double>>& raw, int W, int H,
                    std::vector<BoxD>* out, std::string* err) {
    out->clear();
    for (const auto& b : raw) {
        if (b.size() != 4) { *err = "Each bbox must contain [x1, y1, x2, y2]."; return false; }
        auto clip = [](double v, double hi) { return std::max(0.0, std::min(v, hi)); };
        double x1 = clip(b[0], W), x2 = clip(b[2], W), y1 = clip(b[1], H), y2 = clip(b[3], H);
        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);
        if (!(x2 > x1) || !(y2 > y1)) {
            *err = "bbox must have positive area after clipping";
            return false;
        }
        out->push_back({x1, y1, x2, y2});
    }
    return true;
}

// _order_boxes, mm_bbox_order_mode = "raster": key (x1, y1, caller index).
// O(N log N), N <= a few hundred proposals.
std::vector<int> raster_order(const std::vector<BoxD>& boxes) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (boxes[a][0] != boxes[b][0]) return boxes[a][0] < boxes[b][0];
        if (boxes[a][1] != boxes[b][1]) return boxes[a][1] < boxes[b][1];
        return a < b;
    });
    return order;
}

// re.sub(r"<obj(\d+)>", ...) with an index map; unmapped indices are kept.
std::string remap_obj_tokens(const std::string& s, const std::map<long long, long long>& m) {
    std::string o;
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, 4, "<obj") == 0) {
            size_t j = i + 4;
            while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
            if (j > i + 4 && j < s.size() && s[j] == '>' && j - (i + 4) <= 18) {
                const long long idx = std::stoll(s.substr(i + 4, j - i - 4));
                const auto it = m.find(idx);
                o += it == m.end() ? s.substr(i, j + 1 - i) : "<obj" + std::to_string(it->second) + ">";
                i = j + 1;
                continue;
            }
        }
        o += s[i++];
    }
    return o;
}

struct TaskTemplate { const char* lang; const char* task; const char* tpl; };
const TaskTemplate kTemplates[] = {
#include "task_templates.inc"
};

bool build_task_prompt(const std::string& task, const std::string& lang,
                       const std::string& text, std::string* out) {
    for (const auto& t : kTemplates) {
        if (task != t.task || lang != t.lang) continue;
        const std::string tpl = t.tpl;
        const size_t p = tpl.find("{}");
        *out = tpl.substr(0, p) + text + tpl.substr(p + 2);   // str.format, one slot
        return true;
    }
    return false;
}

// ---------------- official input preprocessing ----------------

// Qwen2VLImageProcessorFast (preprocessor_config.json): smart_resize(factor
// 32, min 65536, max 16777216 px) -> torch uint8 AA bicubic -> (x - 127.5) /
// 127.5 in f32 -> patchify (t duplicated) -> pixel_values; the tower casts to
// bf16. Returns token-major (N, 1536) BEFORE the bf16 cast (= the dumped
// pixel_values) and the bf16 copy fed to the graph.
void primary_pixels(const vlx::ImageU8& img, int* gh, int* gw, std::vector<float>* f32,
                    std::vector<float>* bf16) {
    int hb = 0, wb = 0;
    vlx::smart_resize(img.height, img.width, 32, 65536.0, 16777216.0, &hb, &wb);
    const vlx::ImageU8 rs = vlx::resize_u8(img, wb, hb, vlx::ResampleFilter::Bicubic,
                                           vlx::ResampleImpl::kTorch);
    const int patch = 16, merge = 2, temp = 2;
    *gh = hb / patch;
    *gw = wb / patch;
    const int K = 3 * temp * patch * patch;
    f32->assign(size_t(*gh) * *gw * K, 0.f);
    for (int bh = 0; bh < *gh / merge; ++bh)
        for (int bw = 0; bw < *gw / merge; ++bw)
            for (int mh = 0; mh < merge; ++mh)
                for (int mw = 0; mw < merge; ++mw) {
                    const int p = ((bh * (*gw / merge) + bw) * merge + mh) * merge + mw;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < temp; ++t)
                            for (int iy = 0; iy < patch; ++iy)
                                for (int ix = 0; ix < patch; ++ix) {
                                    const int y = (bh * merge + mh) * patch + iy;
                                    const int x = (bw * merge + mw) * patch + ix;
                                    const size_t k = size_t(c * temp + t) * patch * patch + iy * patch + ix;
                                    const float u = float(rs.rgb[(size_t(y) * wb + x) * 3 + c]);
                                    (*f32)[size_t(p) * K + k] = (u - 127.5f) / 127.5f;
                                }
                }
    bf16->resize(f32->size());
    for (size_t i = 0; i < f32->size(); ++i) (*bf16)[i] = vlx::bf16_round((*f32)[i]);
}

// Aux tower input: CLIPImageProcessor(do_resize=False, rescale only:
// f32(u8) * f32(1/255)) -> .to(bf16) -> _resize_to_supported (nearest
// multiple of 16 per side, banker's rounding; AA bilinear on f32) -> bf16.
// Returns CHW (3, H, W) bf16 values and the HWC copy RadioGraph consumes.
void aux_pixels(const vlx::ImageU8& img, int* H, int* W, std::vector<float>* chw,
                std::vector<float>* hwc) {
    const int h0 = img.height, w0 = img.width;
    const float scale = float(1.0 / 255.0);
    std::vector<float> src(size_t(3) * h0 * w0);
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < size_t(h0) * w0; ++i)
            src[size_t(c) * h0 * w0 + i] = vlx::bf16_round(float(img.rgb[i * 3 + c]) * scale);
    auto nearest16 = [](int v) { return std::max(int(std::nearbyint(v / 16.0)) * 16, 16); };
    *H = nearest16(h0);
    *W = nearest16(w0);
    if (*H == h0 && *W == w0) {
        *chw = std::move(src);
    } else {
        *chw = vlx::resize_f32_bilinear_aa(src, 3, h0, w0, *H, *W);
        for (float& v : *chw) v = vlx::bf16_round(v);
    }
    const size_t n = size_t(*H) * *W;
    hwc->resize(n * 3);
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < n; ++i) (*hwc)[i * 3 + c] = (*chw)[size_t(c) * n + i];
}

// ---------------- llama.cpp helpers ----------------

// Tokenize without BOS, special tokens parsed (HF tokenizer(chunk) on Qwen).
std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    std::vector<llama_token> toks(text.size() + 8);
    const int n = llama_tokenize(vocab, text.data(), int(text.size()), toks.data(),
                                 int(toks.size()), false, true);
    if (n < 0) { std::fprintf(stderr, "tokenize overflow\n"); std::exit(1); }
    toks.resize(size_t(n));
    return toks;
}

std::string token_piece(const llama_vocab* vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, size_t(n)) : std::string();
}

// One prompt entry: a vocab token, or an injected embedding row.
struct Entry {
    llama_token id;        // LLAMA_TOKEN_NULL for embedding rows
    const float* embd;     // n_embd floats when id == LLAMA_TOKEN_NULL
};

// Single llama_process over the whole sequence: tokens get position p,
// embedding rows get the 4 M-RoPE sections {p, p, p, 0} -- exactly what
// llama.cpp expands a text token to, and equal to the official position_ids
// (every section = arange). Logits are requested for the last entry only.
int32_t decode_entries(llama_context* ctx, const std::vector<Entry>& entries, int64_t n_embd,
                       int64_t* pos_cursor) {
    llama_batch_ext* b = llama_batch_ext_init(ctx);
    for (size_t i = 0; i < entries.size(); ++i) {
        const llama_pos p = llama_pos(*pos_cursor + int64_t(i));
        const llama_pos pos4[4] = {p, p, p, 0};
        int32_t idx;
        if (entries[i].id != LLAMA_TOKEN_NULL) {
            idx = llama_batch_ext_add_token(b, 0, entries[i].id);
        } else {
            // n_rows = 1: one batch entry is one position
            idx = llama_batch_ext_add_embd(b, 0, llama_embd{entries[i].embd, 1, size_t(n_embd)});
        }
        if (idx < 0) { llama_batch_ext_free(b); return idx; }
        llama_batch_ext_set_pos(b, idx, pos4);
        if (i + 1 == entries.size()) llama_batch_ext_set_output_logits(b, idx, true);
    }
    const int32_t r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    *pos_cursor += int64_t(entries.size());
    llama_batch_ext_free(b);
    return r;
}

// HF generate token selection: argmax (first max on ties, = torch.argmax)
// when temperature == 0, else temperature -> top-p -> multinomial.
llama_token select_token(const float* logits, int n_vocab, float temperature, float top_p,
                         std::mt19937* rng) {
    int best = 0;
    for (int v = 1; v < n_vocab; ++v)
        if (logits[v] > logits[best]) best = v;
    if (temperature <= 0.f) return best;
    std::vector<std::pair<float, int>> p(static_cast<size_t>(n_vocab));
    double z = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        p[v] = {std::exp((logits[v] - logits[best]) / temperature), v};
        z += p[v].first;
    }
    std::sort(p.begin(), p.end(), [](auto& a, auto& b) { return a.first > b.first; });
    size_t keep = p.size();
    if (top_p < 1.f) {   // TopPLogitsWarper: smallest set with mass >= top_p
        double cum = 0.0;
        for (size_t i = 0; i < p.size(); ++i) {
            cum += p[i].first / z;
            if (cum >= top_p) { keep = i + 1; break; }
        }
    }
    std::vector<double> w(keep);
    for (size_t i = 0; i < keep; ++i) w[i] = p[i].first;
    std::discrete_distribution<size_t> dist(w.begin(), w.end());
    return p[dist(*rng)].second;
}

// Generation loop. Stops AFTER emitting a stop token (python's generate keeps
// it in output_ids, so completion_tokens counts it). Returns the ids.
std::vector<llama_token> generate(llama_context* ctx, const llama_vocab* vocab, int max_new,
                                  const std::vector<llama_token>& stop_ids, float temperature,
                                  float top_p, int64_t* pos_cursor, double* decode_ms) {
    std::vector<llama_token> out;
    std::mt19937 rng(0);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto t0 = Clock::now();
    for (int i = 0; i < max_new; ++i) {
        const llama_token tok = select_token(llama_get_logits_ith(ctx, -1), n_vocab,
                                             temperature, top_p, &rng);
        out.push_back(tok);
        if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end()) break;
        if (i + 1 == max_new) break;
        if (decode_entries(ctx, {{tok, nullptr}}, 0, pos_cursor)) {
            std::fprintf(stderr, "decode failed\n");
            break;
        }
    }
    *decode_ms = ms_since(t0);
    return out;
}

// ---------------- visualization (inference.py::visualize_result_boxes) ----------------

// ImageDraw.rectangle(xy, outline="red", width=3): PIL truncates the float
// corners to int and draws `width` nested 1-px outlines inward; pixels
// outside the image are clipped individually (no corner clamping).
void draw_rect_pil(std::vector<uint8_t>* rgb, int w, int h, double fx0, double fy0, double fx1,
                   double fy1, int width) {
    int x0 = int(fx0), y0 = int(fy0), x1 = int(fx1), y1 = int(fy1);
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);
    auto put = [&](int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        uint8_t* p = rgb->data() + (size_t(y) * w + x) * 3;
        p[0] = 255; p[1] = 0; p[2] = 0;
    };
    for (int i = 0; i < width; ++i) {
        for (int x = x0; x <= x1; ++x) { put(x, y0 + i); put(x, y1 - i); }
        for (int y = y0 + width; y <= y1 - width; ++y) { put(x0 + i, y); put(x1 - i, y); }
    }
}

// 5x7 bitmap font for the label overlay: column bitmaps, bit 0 = top row.
// Zero external font assets; unknown glyphs render as '?'. Non-ASCII labels
// fall back to the <objN> index — the same behavior as the official
// visualize_result_boxes, whose draw.text raises UnicodeEncodeError on
// non-encodable text and retries with result_bbox["object_index"].
const uint8_t kFont5x7[95][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00},
    {0x07, 0x07, 0x00, 0x00, 0x00}, {0x14, 0x7F, 0x14, 0x7F, 0x14},
    {0x24, 0x2A, 0x7F, 0x2A, 0x12}, {0x23, 0x13, 0x08, 0x64, 0x62},
    {0x36, 0x49, 0x55, 0x22, 0x50}, {0x05, 0x03, 0x00, 0x00, 0x00},
    {0x00, 0x1C, 0x22, 0x41, 0x00}, {0x00, 0x41, 0x22, 0x1C, 0x00},
    {0x14, 0x08, 0x3E, 0x08, 0x14}, {0x08, 0x08, 0x3E, 0x08, 0x08},
    {0x00, 0x50, 0x30, 0x00, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08},
    {0x00, 0x60, 0x60, 0x00, 0x00}, {0x20, 0x10, 0x08, 0x04, 0x02},
    {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
    {0x42, 0x61, 0x51, 0x49, 0x46}, {0x21, 0x41, 0x45, 0x4B, 0x31},
    {0x18, 0x14, 0x12, 0x7F, 0x10}, {0x27, 0x45, 0x45, 0x45, 0x39},
    {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
    {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x29, 0x1E},
    {0x00, 0x36, 0x36, 0x00, 0x00}, {0x00, 0x56, 0x36, 0x00, 0x00},
    {0x08, 0x14, 0x22, 0x41, 0x00}, {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x00, 0x41, 0x22, 0x14, 0x08}, {0x02, 0x01, 0x51, 0x09, 0x06},
    {0x32, 0x49, 0x79, 0x41, 0x3E}, {0x7E, 0x11, 0x11, 0x11, 0x7E},
    {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
    {0x7F, 0x41, 0x41, 0x22, 0x1C}, {0x7F, 0x49, 0x49, 0x49, 0x41},
    {0x7F, 0x09, 0x09, 0x09, 0x01}, {0x3E, 0x41, 0x49, 0x49, 0x7A},
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
    {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41},
    {0x7F, 0x40, 0x40, 0x40, 0x40}, {0x7F, 0x02, 0x0C, 0x02, 0x7F},
    {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
    {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E},
    {0x7F, 0x09, 0x19, 0x29, 0x46}, {0x46, 0x49, 0x49, 0x49, 0x31},
    {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
    {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x3F, 0x40, 0x38, 0x40, 0x3F},
    {0x63, 0x14, 0x08, 0x14, 0x63}, {0x07, 0x08, 0x70, 0x08, 0x07},
    {0x61, 0x51, 0x49, 0x45, 0x43}, {0x00, 0x7F, 0x41, 0x41, 0x00},
    {0x02, 0x04, 0x08, 0x10, 0x20}, {0x00, 0x41, 0x41, 0x7F, 0x00},
    {0x04, 0x02, 0x01, 0x02, 0x04}, {0x40, 0x40, 0x40, 0x40, 0x40},
    {0x00, 0x01, 0x02, 0x04, 0x00}, {0x20, 0x54, 0x54, 0x54, 0x78},
    {0x7F, 0x48, 0x44, 0x44, 0x38}, {0x38, 0x44, 0x44, 0x44, 0x20},
    {0x38, 0x44, 0x44, 0x48, 0x7F}, {0x38, 0x54, 0x54, 0x54, 0x18},
    {0x08, 0x7E, 0x09, 0x01, 0x02}, {0x0C, 0x52, 0x52, 0x52, 0x3E},
    {0x7F, 0x08, 0x04, 0x04, 0x78}, {0x00, 0x44, 0x7D, 0x40, 0x00},
    {0x20, 0x40, 0x44, 0x3D, 0x00}, {0x7F, 0x10, 0x28, 0x44, 0x00},
    {0x00, 0x41, 0x7F, 0x40, 0x00}, {0x7C, 0x04, 0x18, 0x04, 0x78},
    {0x7C, 0x08, 0x04, 0x04, 0x78}, {0x38, 0x44, 0x44, 0x44, 0x38},
    {0x7C, 0x14, 0x14, 0x14, 0x08}, {0x08, 0x14, 0x14, 0x18, 0x7C},
    {0x7C, 0x08, 0x04, 0x04, 0x08}, {0x48, 0x54, 0x54, 0x54, 0x20},
    {0x04, 0x3F, 0x44, 0x40, 0x20}, {0x3C, 0x40, 0x40, 0x20, 0x7C},
    {0x1C, 0x20, 0x40, 0x20, 0x1C}, {0x3C, 0x40, 0x30, 0x40, 0x3C},
    {0x44, 0x28, 0x10, 0x28, 0x44}, {0x0C, 0x50, 0x50, 0x50, 0x3C},
    {0x44, 0x64, 0x54, 0x4C, 0x44}, {0x00, 0x08, 0x36, 0x41, 0x00},
    {0x00, 0x00, 0x7F, 0x00, 0x00}, {0x00, 0x41, 0x36, 0x08, 0x00},
    {0x08, 0x08, 0x2A, 0x1C, 0x08},
};

// Red label at (x, y) — official draw.text semantics (top-left anchor). The
// glyph cell is 5x7, doubled on images wider than 640 px so it stays readable
// at the resolutions the pipeline actually sees.
void draw_label_pil(std::vector<uint8_t>* rgb, int w, int h, double fx, double fy,
                    const std::string& label) {
    const int scale = w > 640 ? 2 : 1;
    int x = int(fx) + 2, y = int(fy) + 2;
    for (unsigned char ch : label) {
        if (ch < 0x20 || ch > 0x7E) ch = '?';
        const auto* col = kFont5x7[ch - 0x20];
        for (int cx = 0; cx < 5; ++cx)
            for (int cy = 0; cy < 7; ++cy)
                if (col[cx] & (1u << cy))
                    for (int sy = 0; sy < scale; ++sy)
                        for (int sx = 0; sx < scale; ++sx) {
                            const int px = x + cx * scale + sx, py = y + cy * scale + sy;
                            if (px < 0 || px >= w || py < 0 || py >= h) continue;
                            uint8_t* p = rgb->data() + (size_t(py) * w + px) * 3;
                            p[0] = 255; p[1] = 0; p[2] = 0;
                        }
        x += 6 * scale;
    }
}

static bool all_ascii(const std::string& s) {
    for (unsigned char c : s)
        if (c > 0x7E) return false;
    return true;
}

std::string default_visualization_output(const std::string& image_path) {
    const std::filesystem::path p(image_path);
    return (p.parent_path() / (p.stem().string() + "_result.png")).string();
}

// ---------------- commands ----------------

// cb_eval tap: writes every residual-stream output `l_out-<il>` (f32,
// [n_embd, n_rows]) to <dir>/l_out-<il>.f32 for scripts/compare_llm_layers.py.
bool dump_layer_cb(ggml_tensor* t, bool ask, void* ud) {
    if (std::strncmp(t->name, "l_out-", 6) != 0) return !ask;
    if (ask) return true;
    std::vector<float> v(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    const std::string p = *static_cast<std::string*>(ud) + "/" + t->name + ".f32";
    if (FILE* f = std::fopen(p.c_str(), "wb")) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
    return true;
}

int cmd_gen(int argc, char** argv) {
    std::string model, prompt, embd_file, embd_prefix, embd_suffix, device = "cpu", logits_out;
    std::string layers_dir;
    int n_predict = 48, ngl = -1;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m" || a == "--llm") model = next();
        else if (a == "-p") prompt = next();
        else if (a == "-n") n_predict = std::atoi(next());
        else if (a == "-ngl") ngl = std::atoi(next());
        else if (a == "--device") device = next();
        else if (a == "--embd-file") embd_file = next();       // f32 rows (K, D)
        else if (a == "--embd-prefix") embd_prefix = next();   // text before K rows
        else if (a == "--embd-suffix") embd_suffix = next();   // text after K rows
        else if (a == "--dump-logits") logits_out = next();    // f32 last-position logits
        else if (a == "--dump-layers") layers_dir = next();    // per-layer l_out taps
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model.empty() || (prompt.empty() && embd_file.empty())) {
        std::fprintf(stderr, "gen: -m model (-p prompt | --embd-file ...) [-ngl N]\n");
        return 1;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl >= 0 ? ngl : (device == "cpu" ? 0 : 999);
    llama_model* model_p = llama_model_load_from_file(model.c_str(), mp);
    if (!model_p) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model_p);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    if (!layers_dir.empty()) {
        std::filesystem::create_directories(layers_dir);
        cp.cb_eval = dump_layer_cb;
        cp.cb_eval_user_data = &layers_dir;
    }
    llama_context* ctx = llama_init_from_model(model_p, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }
    const int64_t n_embd = llama_model_n_embd(model_p);

    std::vector<uint8_t> raw;
    std::vector<Entry> entries;
    auto add_text = [&](const std::string& t) {
        for (llama_token id : tokenize(vocab, t)) entries.push_back({id, nullptr});
    };
    size_t rows = 0;
    if (embd_file.empty()) {
        add_text(prompt);
    } else {
        if (!read_file(embd_file, &raw) || raw.empty() || raw.size() % (size_t(n_embd) * 4)) {
            std::fprintf(stderr, "embd file must be f32 rows of n_embd=%lld\n", (long long)n_embd);
            return 1;
        }
        rows = raw.size() / (size_t(n_embd) * 4);
        const float* embd = reinterpret_cast<const float*>(raw.data());
        add_text(embd_prefix);
        for (size_t k = 0; k < rows; ++k) entries.push_back({LLAMA_TOKEN_NULL, embd + k * n_embd});
        add_text(embd_suffix);
    }
    int64_t pos = 0;
    if (entries.empty() || decode_entries(ctx, entries, n_embd, &pos)) {
        std::fprintf(stderr, "decode failed\n");
        return 1;
    }
    if (!logits_out.empty()) {
        // isolates the LLM: feed the official f32 inputs_embeds, compare logits
        FILE* f = std::fopen(logits_out.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", logits_out.c_str()); return 1; }
        std::fwrite(llama_get_logits_ith(ctx, -1), 4, size_t(llama_vocab_n_tokens(vocab)), f);
        std::fclose(f);
    }
    double dms = 0.0;
    std::vector<llama_token> stop = {llama_vocab_eos(vocab)};
    std::string out_text;
    for (llama_token t : generate(ctx, vocab, n_predict, stop, 0.f, 1.f, &pos, &dms))
        out_text += token_piece(vocab, t);
    std::printf("gen ok: %zu injected rows\n---- generation ----\n%s\n-------------------\n",
                rows, out_text.c_str());
    llama_free(ctx);
    llama_model_free(model_p);
    return 0;
}

// detect: WeDetect-Uni proposals straight from the image file
// (inference.py::load_wedetect_proposals). --dump-dir writes the gate taps.
int cmd_detect(int argc, char** argv) {
    std::string model, image, out_json, dump_dir, device = "cpu";
    int threads = default_threads(), repeat = 1;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--wedetect" || a == "-m") model = next();
        else if (a == "--image") image = next();
        else if (a == "--out") out_json = next();
        else if (a == "--dump-dir") dump_dir = next();
        else if (a == "--device") device = next();
        else if (a == "-t" || a == "--threads") threads = std::atoi(next());
        else if (a == "--repeat") repeat = std::max(1, std::atoi(next()));
        else { std::fprintf(stderr, "unknown detect arg: %s\n", a.c_str()); return 1; }
    }
    if (model.empty() || image.empty()) {
        std::fprintf(stderr, "detect: --wedetect W.gguf --image x.jpg [--out boxes.json] "
                             "[--device cpu|cuda|vulkan] [--dump-dir D]\n");
        return 1;
    }
    ggml_backend_t be = init_backend(device, threads);
    if (!be) return 1;
    vlx::GgufFile g;
    std::string err;
    if (!g.load(model, &err)) { std::fprintf(stderr, "gguf: %s\n", err.c_str()); return 1; }
    vlx::ImageU8 img;
    if (!vlx::decode_image(image, &img, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    vlx::WeDetectGraph det(be, &g);
    vlx::WeDetectResult res;
    const bool taps = !dump_dir.empty();
    for (int r = 0; r < repeat; ++r) {
        const auto t0 = Clock::now();
        if (!det.run(img, &res, taps, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::fprintf(stderr, "detect run %d: graph %.1f ms, total %.1f ms\n", r,
                     det.last_graph_ms(), ms_since(t0));
    }
    const std::string js = boxes_to_json(res.boxes);
    if (!out_json.empty() && !write_file(out_json, js.data(), js.size())) return 1;
    if (taps) {
        const vlx::Letterbox lb = vlx::letterbox(img, 640);
        write_file(dump_dir + "/letterbox_u8.bin", lb.canvas.rgb.data(), lb.canvas.rgb.size());
        const char* names[7] = {"c1", "c2", "c3", "c4", "neck0", "neck1", "neck2"};
        for (size_t i = 0; i < res.taps.size(); ++i)
            write_file(dump_dir + "/" + names[i] + ".bin", res.taps[i].data(), res.taps[i].size() * 4);
        write_file(dump_dir + "/cls_logits.bin", res.cls_logits.data(), res.cls_logits.size() * 4);
        write_file(dump_dir + "/bbox_preds.bin", res.bbox_preds.data(), res.bbox_preds.size() * 4);
        std::string sj = "[";
        char buf[48];
        for (size_t i = 0; i < res.scores.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "%s%.9g", i ? ", " : "", double(res.scores[i]));
            sj += buf;
        }
        sj += "]";
        write_file(dump_dir + "/scores.json", sj.data(), sj.size());
    }
    std::printf("detect ok: %zu boxes (%dx%d) -> %s\n", res.boxes.size(), img.width, img.height,
                out_json.empty() ? "(stdout suppressed)" : out_json.c_str());
    ggml_backend_free(be);
    return 0;
}

const char* kRunUsage =
    "run: C++ equivalent of `python inference.py` (same arguments + model files)\n"
    "  --image-path P            [demo/demo_image.jpg]\n"
    "  --task T                  vqa|detection|grounding_single|counting|\n"
    "                            reasoning_detection|brief_region_caption|region_ocr [detection]\n"
    "  --text S                  [\"orange; apple\"]\n"
    "  --lang en|zh              [en]\n"
    "  --bbox-list JSON          '[[x1,y1,x2,y2],...]' (omit -> WeDetect-Uni proposals)\n"
    "  --target-region-indexes J '[2, 3]' (brief_region_caption / region_ocr)\n"
    "  --device cpu|cuda|vulkan  [cpu]      -ngl N  LLM layers offloaded [all on GPU]\n"
    "  --max-new-tokens N [2048] --temperature F [0] --top-p F [1]\n"
    "  --visualize | --no-visualize [on]   --visualization-output P [<image>_result.png]\n"
    "  --warm-runs N             re-run the vision chain N times with weights resident;\n"
    "                            median of the N runs isolates cold start from compute\n"
    "  model files: --gguf-dir D [cpp_ggml/models/gguf] --precision f32|f16|q8_0 [f16]\n"
    "    (D/vlx-vision-10b-<p> merged [mmproj+fpn+radio] when present, else the\n"
    "     per-part vlx-{radio,mmproj,fpn}-10b-<p>; vlx-10b-llm-<p|f16 for f32>;\n"
    "     wedetect-base-uni-f32) or explicitly: --wedetect --radio --mmproj --fpn --llm\n"
    "  -t N threads   --dump-dir D  (gate tensors vs scripts/dump_official_ref.py)\n";

int cmd_run(int argc, char** argv) {
    std::string image_path = "demo/demo_image.jpg", task = "detection", text = "orange; apple";
    std::string lang = "en", bbox_json, target_json, device = "cpu", vis_out, dump_dir;
    std::string gguf_dir = "cpp_ggml/models/gguf", precision = "f16";
    std::string wedetect_m, radio_m, mm_m, fpn_m, llm_m;
    int max_new = 2048, threads = default_threads(), ngl = -1, warm_runs = 0;
    float temperature = 0.f, top_p = 1.f;
    bool visualize = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--image-path" || a == "--image") image_path = next();
        else if (a == "--task") task = next();
        else if (a == "--text") text = next();
        else if (a == "--lang") lang = next();
        else if (a == "--bbox-list") bbox_json = next();
        else if (a == "--target-region-indexes") target_json = next();
        else if (a == "--device") device = next();
        else if (a == "--max-new-tokens") max_new = std::atoi(next().c_str());
        else if (a == "--temperature") temperature = std::strtof(next().c_str(), nullptr);
        else if (a == "--top-p") top_p = std::strtof(next().c_str(), nullptr);
        else if (a == "--visualize") visualize = true;
        else if (a == "--no-visualize") visualize = false;
        else if (a == "--visualization-output") vis_out = next();
        else if (a == "--gguf-dir") gguf_dir = next();
        else if (a == "--precision") precision = next();
        else if (a == "--wedetect" || a == "--detector-checkpoint") wedetect_m = next();
        else if (a == "--radio") radio_m = next();
        else if (a == "--mmproj") mm_m = next();
        else if (a == "--fpn") fpn_m = next();
        else if (a == "--llm") llm_m = next();
        else if (a == "-ngl" || a == "--n-gpu-layers") ngl = std::atoi(next().c_str());
        else if (a == "-t" || a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--warm-runs") warm_runs = std::max(0, std::atoi(next().c_str()));
        else if (a == "--dump-dir") dump_dir = next();
        else if (a == "-h" || a == "--help") { std::fprintf(stderr, "%s", kRunUsage); return 0; }
        else { std::fprintf(stderr, "unknown run arg: %s\n%s", a.c_str(), kRunUsage); return 2; }
    }
    auto pick = [&](std::string* m, const std::string& stem, const std::string& p) {
        if (m->empty()) *m = gguf_dir + "/" + stem + "-" + p + ".gguf";
    };
    // WeDetect stays f32 = the official .pth dtype: its q8_0 changes the box
    // set, and even f16's sub-pixel shifts can reorder near-tied raster keys.
    // The LLM checkpoint is bf16, so f16 already holds it bit-exactly; an f32
    // LLM file would be the same values at twice the bytes.
    pick(&wedetect_m, "wedetect-base-uni", "f32");
    pick(&llm_m, "vlx-10b-llm", precision == "f32" ? "f16" : precision);
    // vision: one merged file per tier when present (scripts/merge_vision_gguf.py);
    // explicit per-part overrides switch back to the three-file layout
    std::string vision_m;
    if (radio_m.empty() && mm_m.empty() && fpn_m.empty()) {
        const std::string cand = gguf_dir + "/vlx-vision-10b-" + precision + ".gguf";
        std::error_code ec;
        if (std::filesystem::exists(cand, ec)) vision_m = cand;
    }
    if (vision_m.empty()) {
        pick(&radio_m, "vlx-radio-10b", precision);
        pick(&mm_m, "vlx-mmproj-10b", precision);
        pick(&fpn_m, "vlx-fpn-10b", precision);
    }

    static const char* kTasks[] = {"vqa", "detection", "grounding_single", "counting",
                                   "reasoning_detection", "brief_region_caption", "region_ocr"};
    if (std::find_if(std::begin(kTasks), std::end(kTasks),
                     [&](const char* t) { return task == t; }) == std::end(kTasks)) {
        std::fprintf(stderr, "invalid --task %s\n", task.c_str());
        return 2;
    }
    const bool region_task = task == "brief_region_caption" || task == "region_ocr";
    if (region_task && target_json.empty()) {
        std::fprintf(stderr, "--target-region-indexes is required for brief_region_caption and "
                             "region_ocr, e.g. --target-region-indexes '[0]'.\n");
        return 2;
    }
    if (lang != "en" && lang != "zh") { std::fprintf(stderr, "invalid --lang %s\n", lang.c_str()); return 2; }

    const auto t_all = Clock::now();
    std::string err;
    vlx::ImageU8 img;
    if (!vlx::decode_image(image_path, &img, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    ggml_backend_t be = init_backend(device, threads);
    if (!be) return 1;
    const double mem_base = double(device_mem_used(device)) / 1073741824.0;   // GiB; others' processes included, so report diffs
    auto dump = [&](const std::string& name, const void* p, size_t bytes) {
        if (!dump_dir.empty()) write_file(dump_dir + "/" + name, p, bytes);
    };

    // ---- proposals (inference.py: every task except vqa) ----
    std::vector<BoxD> caller_boxes;
    bool have_boxes = false;
    if (task != "vqa") {
        std::vector<std::vector<double>> raw;
        if (!bbox_json.empty()) {
            if (!parse_json_numbers(bbox_json, 2, &raw)) {
                std::fprintf(stderr, "--bbox-list must be a JSON list of [x1, y1, x2, y2] boxes.\n");
                return 2;
            }
            std::fprintf(stderr, "Using %zu proposals from --bbox-list.\n", raw.size());
        } else {
            vlx::GgufFile wg;
            if (!wg.load(wedetect_m, &err)) { std::fprintf(stderr, "wedetect gguf: %s\n", err.c_str()); return 1; }
            const auto t0 = Clock::now();
            vlx::WeDetectGraph det(be, &wg);
            vlx::WeDetectResult res;
            if (!det.run(img, &res, false, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
            for (const auto& b : res.boxes) raw.push_back({b.x1, b.y1, b.x2, b.y2});
            std::fprintf(stderr, "[time] wedetect: %zu proposals, graph %.1f ms, total %.1f ms\n",
                         res.boxes.size(), det.last_graph_ms(), ms_since(t0));
        }
        if (!validate_boxes(raw, img.width, img.height, &caller_boxes, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 2;
        }
        have_boxes = true;
    }

    // ---- question (task template + caller -> sorted object index remap) ----
    std::string question;
    if (task == "vqa") {
        question = text;
    } else {
        std::string slot = text;
        if (region_task) {
            std::vector<std::vector<double>> idx;
            if (!parse_json_numbers(target_json, 1, &idx)) {
                std::fprintf(stderr, "--target-region-indexes must be a JSON list of integers.\n");
                return 2;
            }
            if (caller_boxes.empty()) { std::fprintf(stderr, "A region task requires at least one bbox.\n"); return 2; }
            if (idx[0].empty()) { std::fprintf(stderr, "A region task requires at least one target region index.\n"); return 2; }
            slot.clear();
            for (double d : idx[0]) {
                if (d != std::floor(d)) { std::fprintf(stderr, "Target region indexes must be integers.\n"); return 2; }
                if (d < 0 || d >= double(caller_boxes.size())) {
                    std::fprintf(stderr, "Target region index %g is outside bbox_list (expected 0 to %zu).\n",
                                 d, caller_boxes.size() - 1);
                    return 2;
                }
                slot += "<obj" + std::to_string((long long)d) + ">";
            }
        }
        if (!build_task_prompt(task, lang, slot, &question)) {
            std::fprintf(stderr, "no template for task %s / lang %s\n", task.c_str(), lang.c_str());
            return 2;
        }
    }
    const std::vector<int> order = raster_order(caller_boxes);   // sorted -> caller index
    std::map<long long, long long> to_sorted, to_caller;
    for (size_t s = 0; s < order.size(); ++s) {
        to_sorted[order[s]] = long(s);
        to_caller[long(s)] = order[s];
    }
    question = remap_obj_tokens(question, to_sorted);
    // torch.tensor(boxes) -> float32
    std::vector<vlx::Box> boxes(order.size());
    for (size_t s = 0; s < order.size(); ++s) {
        const BoxD& b = caller_boxes[order[s]];
        boxes[s] = {float(b[0]), float(b[1]), float(b[2]), float(b[3])};
    }
    const size_t Nb = boxes.size();

    // ---- vision (each stage frees its device weights before the LLM loads).
    // Repeatable for --warm-runs N: gguf files and graph objects stay resident
    // and only the run() calls repeat, so the median of runs 1..N isolates
    // cold start (weight upload, cuBLAS/cudnn init, graph build) from
    // steady-state compute — the same protocol as python's bench_official.py.
    // Run 0 keeps the original cold-run semantics. ----
    const auto t_vis = Clock::now();
    std::vector<float> image_embeds, object_feats;
    int64_t hidden = 0;
    struct Stage { double prep = 0, mmproj = 0, fpn = 0, radio = 0, hfre = 0, proj = 0, total = 0; };
    std::vector<Stage> stages;
    {
        vlx::GgufFile mguf, fguf, rguf, vguf;
        vlx::GgufFile *mp = &mguf, *fp = &fguf, *rp = &rguf;
        if (!vision_m.empty()) {   // merged vlx-vision-10b-<p>: one read for all three
            if (!vguf.load(vision_m, &err)) { std::fprintf(stderr, "vision gguf: %s\n", err.c_str()); return 1; }
            mp = fp = rp = &vguf;
        } else if (!mguf.load(mm_m, &err)) { std::fprintf(stderr, "mmproj gguf: %s\n", err.c_str()); return 1; }
        vlx::MmprojConfig mc;
        mc.patch_size = int(mp->i32("vlx.mmproj.patch_size", mc.patch_size));
        mc.temporal = int(mp->i32("vlx.mmproj.temporal", mc.temporal));
        mc.embed_dim = int(mp->i32("vlx.mmproj.embed_dim", mc.embed_dim));
        mc.depth = int(mp->i32("vlx.mmproj.depth", mc.depth));
        mc.num_heads = int(mp->i32("vlx.mmproj.num_heads", mc.num_heads));
        mc.mlp_hidden = int(mp->i32("vlx.mmproj.mlp_hidden", mc.mlp_hidden));
        mc.hidden_out = int(mp->i32("vlx.mmproj.hidden_out", mc.hidden_out));
        mc.merge = int(mp->i32("vlx.mmproj.merge", mc.merge));
        mc.pos_side = int(mp->i32("vlx.mmproj.pos_side", mc.pos_side));
        mc.rope_theta = float(mp->f32v("vlx.mmproj.rope_theta", mc.rope_theta));
        hidden = mc.hidden_out;
        vlx::MmprojGraph mm(be, mc, mp);
        vlx::SimpleFpnConfig fc;
        vlx::RadioConfig rc;
        vlx::HfreConfig hc;
        hc.n_threads = threads;
        std::unique_ptr<vlx::SimpleFpnGraph> fpn;
        std::unique_ptr<vlx::RadioGraph> radio;
        if (Nb) {
            if (vision_m.empty() && (!fguf.load(fpn_m, &err) || !rguf.load(radio_m, &err))) {
                std::fprintf(stderr, "fpn/radio gguf: %s\n", err.c_str());
                return 1;
            }
            fc.vt_dim = int(fp->i32("vlx.fpn.vt_dim", mc.hidden_out));
            fc.out_channels = int(fp->i32("vlx.fpn.out_channels", 512));
            fc.roi_output_size = hc.roi_output_size = int(fp->i32("vlx.hfre.roi_output_size", 14));
            fc.roi_aligned = hc.roi_aligned = fp->i32("vlx.hfre.roi_aligned", 1) != 0;
            rc.patch_size = int(rp->i32("vlx.radio.patch_size", rc.patch_size));
            rc.embed_dim = int(rp->i32("vlx.radio.embed_dim", rc.embed_dim));
            rc.depth = int(rp->i32("vlx.radio.depth", rc.depth));
            rc.num_heads = int(rp->i32("vlx.radio.num_heads", rc.num_heads));
            rc.num_cls_tokens = int(rp->i32("vlx.radio.num_cls_tokens", rc.num_cls_tokens));
            rc.num_registers = int(rp->i32("vlx.radio.num_registers", rc.num_registers));
            for (int i = 0; i < 4; ++i)
                rc.out_indices[i] = i == 3 ? rc.depth - 1
                                           : int(std::lround(float(i + 1) * rc.depth / 4.f)) - 1;
            hc.aux_spatial_scale = float(rp->f32v("vlx.hfre.aux_spatial_scale", 1.0 / rc.patch_size));
            fpn.reset(new vlx::SimpleFpnGraph(be, fc, fp));
            radio.reset(new vlx::RadioGraph(be, rc, rp));
        }
        std::fprintf(stderr, "[time] vision setup (gguf reads + parse): %.1f ms\n", ms_since(t_vis));

        for (int w = 0; w <= warm_runs; ++w) {
            Stage st;
            const auto t0 = Clock::now();
            int gh = 0, gw = 0;
            std::vector<float> pix_f32, pix_bf16;
            primary_pixels(img, &gh, &gw, &pix_f32, &pix_bf16);
            st.prep = ms_since(t0);
            if (w == warm_runs) dump("primary_pixels.f32", pix_f32.data(), pix_f32.size() * 4);

            std::vector<float> ll_tokens;
            const auto t1 = Clock::now();
            if (!mm.run(pix_bf16.data(), gh, gw, &ll_tokens, nullptr, &err)) {
                std::fprintf(stderr, "mmproj: %s\n", err.c_str());
                return 1;
            }
            st.mmproj = ms_since(t1);
            // inputs_embeds is bf16 in the official LLM
            image_embeds.resize(ll_tokens.size());
            for (size_t i = 0; i < ll_tokens.size(); ++i) image_embeds[i] = vlx::bf16_round(ll_tokens[i]);

            if (Nb) {
                // vt branch: SimpleFPN over the merged main-tower feature
                std::vector<vlx::FeatureLevel> vt_levels;
                const auto t2 = Clock::now();
                const bool roialign_gpu =
#if defined(VLX_HAS_ROIALIGN_CUDA)
                    device == "cuda";
#else
                    false;
#endif
                if (!fpn->run(ll_tokens.data(), gh, gw, &vt_levels, &err, roialign_gpu)) {
                    std::fprintf(stderr, "fpn run failed\n");
                    return 1;
                }
                st.fpn = ms_since(t2);
                // aux branch: C-RADIOv4 on the round-16 resized image
                int aH = 0, aW = 0;
                std::vector<float> aux_chw, aux_hwc;
                aux_pixels(img, &aH, &aW, &aux_chw, &aux_hwc);
                if (w == warm_runs) dump("aux_input.f32", aux_chw.data(), aux_chw.size() * 4);
                std::vector<vlx::FeatureLevel> aux_levels;
                const auto t3 = Clock::now();
                if (!radio->run(aux_hwc.data(), aH, aW, &aux_levels, &err, roialign_gpu)) {
                    std::fprintf(stderr, "radio: %s\n", err.c_str());
                    return 1;
                }
                st.radio = ms_since(t3);
                // encode_objects: aux RoIs use the caller-pixel boxes; the vt
                // branch and the position embedding use vt_boxes = boxes * (vt
                // input size / original size), scale computed in python floats
                // then torch.tensor -> float32.
                const int hb = gh * 16, wb = gw * 16;
                const float sw = float(double(wb) / img.width), sh = float(double(hb) / img.height);
                std::vector<vlx::Box> vt_boxes(Nb);
                for (size_t n = 0; n < Nb; ++n)
                    vt_boxes[n] = {boxes[n].x1 * sw, boxes[n].y1 * sh, boxes[n].x2 * sw, boxes[n].y2 * sh};
                vlx::HfreEncoder hfre(hc);
                const auto t4 = Clock::now();
                std::vector<float> aux_nxc, vt_nxc, fused;
                bool pooled = false;
#if defined(VLX_HAS_ROIALIGN_CUDA)
                if (roialign_gpu) {
                    ggml_tensor* aux_ts[4];
                    ggml_tensor* vt_ts[4];
                    for (int li = 0; li < 4; ++li) {
                        aux_ts[li] = radio->out_level_gpu(li);
                        vt_ts[li] = fpn->out_level_gpu(li);
                    }
                    const std::vector<float> strides = {4.f, 8.f, 16.f, 32.f};
                    std::string cerr2;
                    float* d_aux_out = nullptr;
                    float* d_vt_out[4] = {};
                    if ((cerr2 = vlx::cuda_encode_aux_dev(aux_ts, 4, boxes, hc.roi_output_size,
                                                          hc.roi_aligned, hc.aux_spatial_scale,
                                                          nullptr, &d_aux_out)).empty()) {
                        const int Cvt = vt_levels[0].channels;
                        const int aux_C = int(aux_ts[0]->ne[2]) * 4;
                        pooled = true;
                        for (int li = 0; li < 4 && pooled; ++li) {
                            if ((cerr2 = vlx::cuda_pool_dev(vt_ts[li], vt_boxes, 1.f / strides[li],
                                                            hc.roi_output_size, hc.roi_aligned,
                                                            nullptr, &d_vt_out[li])).empty()) {
                            } else {
                                pooled = false;
                            }
                        }
                        if (pooled) {
                            // fused HFRE fusion on the GPU: concat + sine pos-embed,
                            // frame extents identical to vt_grid_ref[0] below
                            const float img_w = float((gw / 2) * int(1.f / hc.vt_spatial_scale));
                            const float img_h = float((gh / 2) * int(1.f / hc.vt_spatial_scale));
                            if ((cerr2 = vlx::cuda_fuse_dev(d_aux_out, d_vt_out[0], d_vt_out[1],
                                                            d_vt_out[2], d_vt_out[3], boxes,
                                                            img_w, img_h, aux_C, Cvt,
                                                            hc.pos_embedding_dim,
                                                            &fused)).empty()) {
                                vt_nxc.resize(0);  // fused directly; vt staging unused
                            } else {
                                pooled = false;
                            }
                        }
                    }
                    const float* cleanup[5] = {d_aux_out, d_vt_out[0], d_vt_out[1],
                                               d_vt_out[2], d_vt_out[3]};
                    vlx::cuda_free_dev_ptrs(cleanup, 5);
                    if (!pooled)
                        std::fprintf(stderr, "roi-align cuda fallback: %s\n", cerr2.c_str());
                }
#endif
                if (!pooled) {
#if defined(VLX_HAS_ROIALIGN_CUDA)
                    if (roialign_gpu) {
                        // lazy host read-back of the device-resident levels
                        for (int li = 0; li < 4; ++li) {
                            ggml_tensor* t = radio->out_level_gpu(li);
                            aux_levels[li].data.resize(ggml_nbytes(t) / sizeof(float));
                            ggml_backend_tensor_get(t, aux_levels[li].data.data(), 0,
                                                    ggml_nbytes(t));
                            t = fpn->out_level_gpu(li);
                            vt_levels[li].data.resize(ggml_nbytes(t) / sizeof(float));
                            ggml_backend_tensor_get(t, vt_levels[li].data.data(), 0,
                                                    ggml_nbytes(t));
                        }
                    }
#endif
                    if (!hfre.encode_aux(boxes, aux_levels, &aux_nxc, &err) ||
                        !hfre.encode_vt(vt_boxes, vt_levels, {4.f, 8.f, 16.f, 32.f}, &vt_nxc, &err)) {
                        std::fprintf(stderr, "hfre: %s\n", err.c_str());
                        return 1;
                    }
                }
                // pos-embed frame = vt feature extent / vt_spatial_scale = (h_bar, w_bar)
                std::vector<vlx::FeatureLevel> vt_grid_ref(1);
                vt_grid_ref[0].width = (gw / 2) * int(1.f / hc.vt_spatial_scale);
                vt_grid_ref[0].height = (gh / 2) * int(1.f / hc.vt_spatial_scale);
                if (!pooled &&
                    !hfre.fuse(aux_nxc, vt_nxc, vt_boxes, int(aux_nxc.size() / Nb), int(vt_nxc.size() / Nb),
                               vt_grid_ref, &fused, &err)) {
                    std::fprintf(stderr, "hfre fuse: %s\n", err.c_str());
                    return 1;
                }
                st.hfre = ms_since(t4);
                for (float& v : fused) v = vlx::bf16_round(v);   // .to(mm_projector_aux dtype)
                if (w == warm_runs) dump("hfre_fused.f32", fused.data(), fused.size() * 4);

                const auto t5 = Clock::now();
                if (!vlx::run_mlp2x_gelu(be, *fp, "aux_proj", fused, int64_t(Nb), &object_feats)) {
                    std::fprintf(stderr, "aux_proj projector failed\n");
                    return 1;
                }
                st.proj = ms_since(t5);
                for (float& v : object_feats) v = vlx::bf16_round(v);
            }
            st.total = ms_since(t0);
            stages.push_back(st);
            std::fprintf(stderr, "[time] vision run %d%s: prep %.1f | mmproj %.1f | fpn %.1f | "
                                 "radio %.1f | hfre %.1f | proj %.1f | total %.1f ms\n",
                         w, w == 0 ? " (cold)" : "", st.prep, st.mmproj, st.fpn, st.radio,
                         st.hfre, st.proj, st.total);
        }
    }
    if (warm_runs > 0) {
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            const size_t n = v.size();
            return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
        };
        const double Stage::* fld[7] = {&Stage::prep, &Stage::mmproj, &Stage::fpn, &Stage::radio,
                                        &Stage::hfre, &Stage::proj, &Stage::total};
        double m[7];
        for (int k = 0; k < 7; ++k) {
            std::vector<double> col;
            for (size_t s = 1; s < stages.size(); ++s) col.push_back(stages[s].*fld[k]);
            m[k] = med(col);
        }
        std::fprintf(stderr, "[time] vision warm median of %d: prep %.1f | mmproj %.1f | fpn %.1f | "
                             "radio %.1f | hfre %.1f | proj %.1f | total %.1f ms\n",
                     warm_runs, m[0], m[1], m[2], m[3], m[4], m[5], m[6]);
    }
    const double mem_vision = double(device_mem_used(device)) / 1073741824.0;
    std::fprintf(stderr, "[time] vision total: %.1f ms\n", ms_since(t_vis));
    ggml_backend_free(be);
    dump("image_embeds.f32", image_embeds.data(), image_embeds.size() * 4);
    dump("object_feats.f32", object_feats.data(), object_feats.size() * 4);

    // ---- LLM ----
    const auto t_load = Clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl >= 0 ? ngl : (device == "cpu" ? 0 : 999);
    llama_model* model_p = llama_model_load_from_file(llm_m.c_str(), mp);
    if (!model_p) { std::fprintf(stderr, "llm load failed: %s\n", llm_m.c_str()); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model_p);
    const int64_t n_embd = llama_model_n_embd(model_p);
    if (n_embd != hidden) {
        std::fprintf(stderr, "LLM n_embd %lld != vision hidden %lld\n", (long long)n_embd, (long long)hidden);
        return 1;
    }

    // _build_prompt + tokenizer_image_object_token: split on <image>, then
    // <objfeat>, tokenize every chunk separately (no BOS for Qwen).
    std::string prompt_img = "<|vision_start|><image><|vision_end|>";
    if (Nb) {
        prompt_img += "\n";
        for (size_t i = 0; i < Nb; ++i) prompt_img += "<obj" + std::to_string(i) + "><objfeat>";
    }
    const std::string prompt = "<|im_start|>user\n" + prompt_img + "\n" + question +
                               "<|im_end|>\n<|im_start|>assistant\n";
    const size_t n_img = image_embeds.size() / size_t(n_embd);
    std::vector<Entry> entries;
    std::vector<int32_t> ids;   // official expanded input_ids (dump)
    size_t img_seen = 0, obj_seen = 0;
    auto split = [](const std::string& s, const std::string& sep) {
        std::vector<std::string> parts;
        size_t p = 0, q;
        while ((q = s.find(sep, p)) != std::string::npos) { parts.push_back(s.substr(p, q - p)); p = q + sep.size(); }
        parts.push_back(s.substr(p));
        return parts;
    };
    const auto img_chunks = split(prompt, "<image>");
    for (size_t ci = 0; ci < img_chunks.size(); ++ci) {
        const auto obj_chunks = Nb ? split(img_chunks[ci], "<objfeat>")
                                   : std::vector<std::string>{img_chunks[ci]};
        for (size_t oi = 0; oi < obj_chunks.size(); ++oi) {
            if (oi > 0) {
                if (obj_seen >= Nb) { std::fprintf(stderr, "more <objfeat> than boxes\n"); return 1; }
                entries.push_back({LLAMA_TOKEN_NULL, object_feats.data() + obj_seen++ * n_embd});
                ids.push_back(248181);
            }
            for (llama_token t : tokenize(vocab, obj_chunks[oi])) {
                entries.push_back({t, nullptr});
                ids.push_back(t);
            }
        }
        if (ci + 1 < img_chunks.size()) {
            if (img_seen++) { std::fprintf(stderr, "Prompt contains more image tokens than input images.\n"); return 1; }
            for (size_t r = 0; r < n_img; ++r) {
                entries.push_back({LLAMA_TOKEN_NULL, image_embeds.data() + r * n_embd});
                ids.push_back(248056);
            }
        }
    }
    dump("input_ids.i32", ids.data(), ids.size() * 4);

    llama_context_params cp = llama_context_default_params();
    const uint32_t L = uint32_t(entries.size());
    cp.n_ctx = ((L + uint32_t(std::max(max_new, 1)) + 255) / 256) * 256;
    cp.n_batch = std::max<uint32_t>(L, 512);
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    llama_context* lctx = llama_init_from_model(model_p, cp);
    if (!lctx) { std::fprintf(stderr, "llm context failed\n"); return 1; }
    const double mem_llm = double(device_mem_used(device)) / 1073741824.0;
    const double load_ms = ms_since(t_load);

    const auto t_pre = Clock::now();
    int64_t pos = 0;
    if (decode_entries(lctx, entries, n_embd, &pos)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
    llama_synchronize(lctx);  // llama_decode is async on GPU backends; time the work, not the enqueue
    const double prefill_ms = ms_since(t_pre);
    if (!dump_dir.empty())
        dump("step0_logits.f32", llama_get_logits_ith(lctx, -1), size_t(llama_vocab_n_tokens(vocab)) * 4);

    // stopping: KeywordsStoppingCriteria(["<|im_end|>"]) + generation_config eos
    std::vector<llama_token> stop = tokenize(vocab, "<|im_end|>");
    if (stop.size() != 1) { std::fprintf(stderr, "<|im_end|> is not a single token\n"); return 1; }
    stop.push_back(llama_vocab_eos(vocab));
    double decode_ms = 0.0;
    const std::vector<llama_token> gen =
        generate(lctx, vocab, max_new, stop, temperature, top_p, &pos, &decode_ms);
    std::fprintf(stderr, "[time] llm load %.1f ms | prefill %u tok %.1f ms | decode %zu tok %.1f ms "
                         "(%.2f tok/s) | total %.1f ms | dev-mem %.1f GiB\n",
                 load_ms, L, prefill_ms, gen.size(), decode_ms,
                 gen.size() > 1 ? (gen.size() - 1) * 1000.0 / decode_ms : 0.0, ms_since(t_all),
                 std::max(mem_vision, mem_llm) - mem_base);
    if (!dump_dir.empty()) {
        std::string sj = "[";
        for (size_t i = 0; i < gen.size(); ++i) sj += (i ? ", " : "") + std::to_string(gen[i]);
        sj += "]";
        dump("step_argmax.json", sj.data(), sj.size());
    }

    // tokenizer.decode(skip_special_tokens=False) -> drop <|im_end|> -> strip
    std::string answer;
    for (llama_token t : gen) answer += token_piece(vocab, t);
    for (size_t p; (p = answer.find("<|im_end|>")) != std::string::npos;) answer.erase(p, 10);
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    answer.erase(answer.begin(), std::find_if(answer.begin(), answer.end(), not_space));
    answer.erase(std::find_if(answer.rbegin(), answer.rend(), not_space).base(), answer.end());
    answer = remap_obj_tokens(answer, to_caller);

    // _build_result_bbox_list
    struct ResultBox { std::string index, label; BoxD box; };
    std::vector<ResultBox> results;
    if (!caller_boxes.empty()) {
        static const std::regex grp(R"(<ground>([\s\S]*?)</ground>\s*<objects>([\s\S]*?)</objects>)");
        static const std::regex obj(R"(<obj(\d+)>)");
        for (std::sregex_iterator it(answer.begin(), answer.end(), grp), end; it != end; ++it) {
            std::string label = (*it)[1].str();
            label.erase(label.begin(), std::find_if(label.begin(), label.end(), not_space));
            label.erase(std::find_if(label.rbegin(), label.rend(), not_space).base(), label.end());
            const std::string objs = (*it)[2].str();
            for (std::sregex_iterator jt(objs.begin(), objs.end(), obj); jt != end; ++jt) {
                const std::string digits = (*jt)[1].str();
                if (digits.size() > 18) continue;
                const long long k = std::stoll(digits);
                if (k >= (long long)caller_boxes.size()) continue;
                results.push_back({(*jt)[0].str(), label, caller_boxes[size_t(k)]});
            }
        }
    }

    std::string vis_path;
    if (visualize && !results.empty()) {
        vis_path = vis_out.empty() ? default_visualization_output(image_path) : vis_out;
        std::vector<uint8_t> canvas = img.rgb;
        for (const auto& r : results) {
            draw_rect_pil(&canvas, img.width, img.height, r.box[0], r.box[1], r.box[2], r.box[3], 3);
            // official visualize_result_boxes draws the label at (xmin+2, ymin+2)
            // and retries with the <objN> index when the text is not encodable
            const std::string& label = all_ascii(r.label) ? r.label : r.index;
            draw_label_pil(&canvas, img.width, img.height, r.box[0], r.box[1], label);
        }
        if (!stbi_write_png(vis_path.c_str(), img.width, img.height, 3, canvas.data(), img.width * 3)) {
            std::fprintf(stderr, "visualization write failed: %s\n", vis_path.c_str());
            vis_path.clear();
        }
    }

    // print(json.dumps(result, ensure_ascii=False, indent=2))
    std::string js = "{\n  \"answer\": " + py_json_str(answer) + ",\n  \"result_bbox_list\": ";
    if (results.empty()) {
        js += "[]";
    } else {
        js += "[\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            js += "    {\n      \"object_index\": " + py_json_str(r.index) +
                  ",\n      \"xmin\": " + py_float(r.box[0]) + ",\n      \"ymin\": " + py_float(r.box[1]) +
                  ",\n      \"xmax\": " + py_float(r.box[2]) + ",\n      \"ymax\": " + py_float(r.box[3]) +
                  ",\n      \"label\": " + py_json_str(r.label) + "\n    }" +
                  (i + 1 < results.size() ? ",\n" : "\n");
        }
        js += "  ]";
    }
    js += ",\n  \"prompt_tokens\": " + std::to_string(L) +
          ",\n  \"completion_tokens\": " + std::to_string(gen.size());
    if (!vis_path.empty()) js += ",\n  \"visualization_path\": " + py_json_str(vis_path);
    js += "\n}\n";
    std::fputs(js.c_str(), stdout);
    dump("result.json", js.data(), js.size());

    llama_free(lctx);
    llama_model_free(model_p);
    return 0;
}

// visualize: draw a boxes.json on an image with the inference.py rectangle.
int cmd_visualize(int argc, char** argv) {
    std::string image_path, boxes_json, out_path;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--image") image_path = next();
        else if (a == "--boxes") boxes_json = next();
        else if (a == "--out") out_path = next();
        else { std::fprintf(stderr, "unknown visualize arg: %s\n", a.c_str()); return 1; }
    }
    if (image_path.empty() || boxes_json.empty()) {
        std::fprintf(stderr, "visualize: --image x.jpg --boxes boxes.json [--out out.png]\n");
        return 1;
    }
    vlx::ImageU8 img;
    std::string err;
    if (!vlx::decode_image(image_path, &img, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<std::vector<double>> boxes;
    if (!parse_json_numbers(read_text(boxes_json), 2, &boxes)) {
        std::fprintf(stderr, "boxes.json parse failed\n");
        return 1;
    }
    for (const auto& b : boxes)
        if (b.size() == 4) draw_rect_pil(&img.rgb, img.width, img.height, b[0], b[1], b[2], b[3], 3);
    const std::string out = out_path.empty() ? default_visualization_output(image_path) : out_path;
    if (!stbi_write_png(out.c_str(), img.width, img.height, 3, img.rgb.data(), img.width * 3)) {
        std::fprintf(stderr, "stbi_write_png failed\n");
        return 1;
    }
    std::printf("visualize ok: %zu boxes -> %s\n", boxes.size(), out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf(
            "vlx-cli — VLX-Seek-1.5 C++ full chain (ggml vision + llama.cpp LLM)\n\n"
            "  run       [inference.py arguments]   (run --help)\n"
            "  detect    --wedetect W.gguf --image x.jpg [--out boxes.json] [--device D]\n"
            "  gen       -m LLM.gguf -p prompt [-n N] [-ngl N]\n"
            "  gen-embd  -m LLM.gguf --embd-file f32(K,D) --embd-prefix T --embd-suffix T [-n N]\n"
            "  visualize --image x.jpg --boxes boxes.json [--out out.png]\n");
        return 1;
    }
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
    }, nullptr);
    const std::string cmd = argv[1];
    if (cmd == "run") return cmd_run(argc, argv);
    if (cmd == "gen" || cmd == "gen-embd") return cmd_gen(argc, argv);
    if (cmd == "detect") return cmd_detect(argc, argv);
    if (cmd == "visualize") return cmd_visualize(argc, argv);
    std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
    return 1;
}
