// radio-cli: vision sub-chain driver for development and parity validation.
//
// Commands:
//   radio  --model <gguf> (--input-bin <f32 HWC> | --image <jpg/png>)
//                 [--height H --width W] [--dump <prefix>] [--hfre boxes.json]
//
// The --input-bin path consumes the exact preprocessed tensor that
// scripts/dump_torch_stages.py writes (float32 HWC, [0,1]), so C++ vs torch
// parity is measured on identical input bytes end to end. Direct image
// loading is wired once stb_image is vendored (P3).
#include "ggml_util.hpp"
#include "gguf_loader.hpp"
#include "hfre.hpp"
#include "mmproj_graph.hpp"
#include "radio_graph.hpp"
#include "simplefpn_graph.hpp"
#include "vlx_types.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// ggml v0.21: CPU backend init goes through the device registry.
ggml_backend_t init_cpu_backend() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) return nullptr;
    return ggml_backend_dev_init(dev, nullptr);
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

bool write_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(data), std::streamsize(bytes));
    return true;
}

void dump_f32(const std::string& prefix, const std::string& name,
              const void* data, size_t bytes) {
    write_file(prefix + "_" + name + ".bin", data, bytes);
}

void print_usage() {
    std::printf(
        "radio-cli — VLX-Seek vision sub-chain (C-RADIOv4 + Qwen3.5-VL mmproj + HFRE)\n\n"
        "  radio-cli radio --model m.gguf --input-bin in.bin --height H --width W\n"
        "        [--dump PREFIX] [--hfre boxes.json] [--boxes-img-w W --boxes-img-h H]\n\n"
        "  radio-cli mmproj --model m.gguf --input-bin patches.bin --grid-h H --grid-w W\n"
        "        [--dump-llm out.bin] [--dump-final out.bin]\n"
        "        [--fpn fpn.gguf --hfre boxes.json [--dump-fpn PREFIX] [--dump-vt out.bin]]\n\n"
        "  radio --input-bin : float32 HWC (rows, channels-last), values in [0,1]\n"
        "                (byte-identical to dump_torch_stages.py output)\n"
        "  radio --dump  : writes PREFIX_radio_L{0..3}.bin (NCHW f32) and, with\n"
        "                --hfre, PREFIX_hfre_obj.bin (N x D f32)\n"
        "  mmproj --input-bin : float32 (N, 3*2*16*16) patch bytes in spatial-\n"
        "                merge-block order (byte-identical to dump_torch_mmproj.py\n"
        "                input.bin); N = grid_h*grid_w\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 1; }
    const std::string cmd = argv[1];
    if (cmd == "diag") {
        // diag --model M --diag-input postcls.bin --height H --width W
        //      --point n1|qkv|attnbr|mlpbr|x --out out.bin
        std::string model, din, point, outf;
        int H = 0, W = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
            if (a == "--model") model = next();
            else if (a == "--diag-input") din = next();
            else if (a == "--height") H = std::atoi(next());
            else if (a == "--width") W = std::atoi(next());
            else if (a == "--point") point = next();
            else if (a == "--out") outf = next();
        }
        if (model.empty() || din.empty() || !H || !W || point.empty() || outf.empty()) {
            std::fprintf(stderr, "diag: --model --diag-input --height --width --point --out required\n");
            return 1;
        }
        vlx::GgufFile gguf;
        std::string err;
        if (!gguf.load(model, &err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        vlx::RadioConfig rc;
        rc.patch_size = int(gguf.i32("vlx.radio.patch_size", rc.patch_size));
        rc.embed_dim = int(gguf.i32("vlx.radio.embed_dim", rc.embed_dim));
        rc.depth = int(gguf.i32("vlx.radio.depth", rc.depth));
        rc.num_heads = int(gguf.i32("vlx.radio.num_heads", rc.num_heads));
        rc.num_cls_tokens = int(gguf.i32("vlx.radio.num_cls_tokens", rc.num_cls_tokens));
        rc.num_registers = int(gguf.i32("vlx.radio.num_registers", rc.num_registers));
        ggml_backend_t be = init_cpu_backend();
        vlx::RadioGraph radio(be, rc, &gguf);
        const size_t n_elem = size_t(rc.num_skip() + (H / rc.patch_size) * (W / rc.patch_size)) * rc.embed_dim;
        std::vector<uint8_t> raw;
        if (!read_file(din, &raw) || raw.size() != n_elem * 4) {
            std::fprintf(stderr, "error: diag-input must be %zu f32 values\n", n_elem);
            return 1;
        }
        std::vector<float> outv;
        if (!radio.run_diag(reinterpret_cast<const float*>(raw.data()), point.c_str(),
                            H, W, &outv, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        if (!write_file(outf, outv.data(), outv.size() * 4)) return 1;
        std::printf("diag %s ok: %zu floats -> %s\n", point.c_str(), outv.size(), outf.c_str());
        ggml_backend_free(be);
        return 0;
    }

    if (cmd == "mmproj-diag") {
        // mmproj-diag --model M --input-bin patches.bin --grid-h H --grid-w W
        //             --point n1|qkv|q_rot|attnbr|mlpbr|x --out out.bin
        std::string model, din, point, outf;
        int gh = 0, gw = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
            if (a == "--model") model = next();
            else if (a == "--input-bin") din = next();
            else if (a == "--grid-h") gh = std::atoi(next());
            else if (a == "--grid-w") gw = std::atoi(next());
            else if (a == "--point") point = next();
            else if (a == "--out") outf = next();
        }
        if (model.empty() || din.empty() || gh <= 0 || gw <= 0 || point.empty() || outf.empty()) {
            std::fprintf(stderr, "mmproj-diag: --model/--input-bin/--grid-h/--grid-w/--point/--out required\n");
            return 1;
        }
        ggml_backend_t be = init_cpu_backend();
        if (!be) { std::fprintf(stderr, "no CPU backend\n"); return 1; }
        vlx::GgufFile gguf;
        std::string gerr;
        if (!gguf.load(model, &gerr)) {
            std::fprintf(stderr, "gguf load failed: %s\n", gerr.c_str());
            return 1;
        }
        vlx::MmprojConfig mc;
        mc.patch_size = int(gguf.i32("vlx.mmproj.patch_size", mc.patch_size));
        mc.temporal = int(gguf.i32("vlx.mmproj.temporal", mc.temporal));
        mc.embed_dim = int(gguf.i32("vlx.mmproj.embed_dim", mc.embed_dim));
        mc.depth = int(gguf.i32("vlx.mmproj.depth", mc.depth));
        mc.num_heads = int(gguf.i32("vlx.mmproj.num_heads", mc.num_heads));
        mc.mlp_hidden = int(gguf.i32("vlx.mmproj.mlp_hidden", mc.mlp_hidden));
        mc.hidden_out = int(gguf.i32("vlx.mmproj.hidden_out", mc.hidden_out));
        mc.merge = int(gguf.i32("vlx.mmproj.merge", mc.merge));
        mc.pos_side = int(gguf.i32("vlx.mmproj.pos_side", mc.pos_side));
        mc.rope_theta = float(gguf.f32v("vlx.mmproj.rope_theta", mc.rope_theta));
        vlx::MmprojGraph mm(be, mc, &gguf);
        const size_t N = size_t(gh) * gw;
        std::vector<uint8_t> raw;
        if (!read_file(din, &raw) || raw.size() != N * mc.kdim() * 4) {
            std::fprintf(stderr, "error: input must be %zu f32 values\n", N * mc.kdim());
            return 1;
        }
        std::vector<float> outv;
        if (!mm.run_diag(reinterpret_cast<const float*>(raw.data()), gh, gw,
                         point.c_str(), &outv)) {
            std::fprintf(stderr, "diag failed (bad point?)\n");
            return 1;
        }
        if (!write_file(outf, outv.data(), outv.size() * 4)) return 1;
        std::printf("mmproj-diag %s ok: %zu floats -> %s\n", point.c_str(), outv.size(), outf.c_str());
        ggml_backend_free(be);
        return 0;
    }

    if (cmd == "mmproj") {
        // mmproj --model M --input-bin patches.bin --grid-h H --grid-w W
        //        [--dump-llm out.bin] [--dump-final out.bin]
        //        [--fpn fpn.gguf --hfre boxes.json [--dump-fpn P] [--dump-vt o]]
        std::string model, din, dump_llm, dump_final;
        std::string fpn_model, hfre_json, dump_fpn, dump_vt;
        int gh = 0, gw = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
            if (a == "--model") model = next();
            else if (a == "--input-bin") din = next();
            else if (a == "--grid-h") gh = std::atoi(next());
            else if (a == "--grid-w") gw = std::atoi(next());
            else if (a == "--dump-llm") dump_llm = next();
            else if (a == "--dump-final") dump_final = next();
            else if (a == "--fpn") fpn_model = next();
            else if (a == "--hfre") hfre_json = next();
            else if (a == "--dump-fpn") dump_fpn = next();
            else if (a == "--dump-vt") dump_vt = next();
            else { std::fprintf(stderr, "unknown mmproj arg: %s\n", a.c_str()); return 1; }
        }
        if (model.empty() || din.empty() || gh <= 0 || gw <= 0) {
            std::fprintf(stderr, "mmproj: --model/--input-bin/--grid-h/--grid-w required\n");
            return 1;
        }
        ggml_backend_t be = init_cpu_backend();
        if (!be) { std::fprintf(stderr, "no CPU backend\n"); return 1; }

        vlx::GgufFile gguf;
        std::string gerr;
        if (!gguf.load(model, &gerr)) {
            std::fprintf(stderr, "gguf load failed: %s\n", gerr.c_str());
            return 1;
        }
        vlx::MmprojConfig mc;
        mc.patch_size = int(gguf.i32("vlx.mmproj.patch_size", mc.patch_size));
        mc.temporal = int(gguf.i32("vlx.mmproj.temporal", mc.temporal));
        mc.embed_dim = int(gguf.i32("vlx.mmproj.embed_dim", mc.embed_dim));
        mc.depth = int(gguf.i32("vlx.mmproj.depth", mc.depth));
        mc.num_heads = int(gguf.i32("vlx.mmproj.num_heads", mc.num_heads));
        mc.mlp_hidden = int(gguf.i32("vlx.mmproj.mlp_hidden", mc.mlp_hidden));
        mc.hidden_out = int(gguf.i32("vlx.mmproj.hidden_out", mc.hidden_out));
        mc.merge = int(gguf.i32("vlx.mmproj.merge", mc.merge));
        mc.pos_side = int(gguf.i32("vlx.mmproj.pos_side", mc.pos_side));
        mc.rope_theta = float(gguf.f32v("vlx.mmproj.rope_theta", mc.rope_theta));
        std::printf("mmproj: dim%d depth%d heads%d mlp%d pos%dx%d theta%g\n",
                    mc.embed_dim, mc.depth, mc.num_heads, mc.mlp_hidden,
                    mc.pos_side, mc.pos_side, mc.rope_theta);

        vlx::MmprojGraph mm(be, mc, &gguf);
        const size_t N = size_t(gh) * gw;
        std::vector<uint8_t> raw;
        if (!read_file(din, &raw) || raw.size() != N * mc.kdim() * 4) {
            std::fprintf(stderr, "error: input must be %zu f32 values\n", N * mc.kdim());
            return 1;
        }
        std::vector<float> ll_tokens, final_tokens;
        if (!mm.run(reinterpret_cast<const float*>(raw.data()), gh, gw,
                    &ll_tokens, &final_tokens)) {
            std::fprintf(stderr, "mmproj run failed\n");
            return 1;
        }
        if (!dump_final.empty() && !final_tokens.empty())
            write_file(dump_final, final_tokens.data(), final_tokens.size() * 4);
        if (!dump_llm.empty() && !ll_tokens.empty())
            write_file(dump_llm, ll_tokens.data(), ll_tokens.size() * 4);

        // ---- HFRE vt branch: SimpleFPN over the pooler output ----
        if (!fpn_model.empty()) {
            vlx::GgufFile fguf;
            std::string ferr;
            if (!fguf.load(fpn_model, &ferr)) {
                std::fprintf(stderr, "fpn gguf load failed: %s\n", ferr.c_str());
                return 1;
            }
            vlx::SimpleFpnConfig fc;
            fc.vt_dim = int(fguf.i32("vlx.fpn.vt_dim", mc.hidden_out));
            fc.out_channels = int(fguf.i32("vlx.fpn.out_channels", 512));
            fc.roi_output_size = int(fguf.i32("vlx.hfre.roi_output_size", 14));
            fc.roi_aligned = fguf.i32("vlx.hfre.roi_aligned", 1) != 0;
            vlx::SimpleFpnGraph fpn(be, fc, &fguf);
            std::vector<vlx::FeatureLevel> levels;
            if (!fpn.run(ll_tokens.data(), gh, gw, &levels)) {
                std::fprintf(stderr, "fpn run failed\n");
                return 1;
            }
            std::printf("fpn ok: %zu levels, finest %dx%d c%d\n", levels.size(),
                        levels[0].height, levels[0].width, levels[0].channels);
            if (!dump_fpn.empty())
                for (size_t li = 0; li < levels.size(); ++li) {
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s_fpn_L%zu.bin",
                                  dump_fpn.c_str(), li);
                    write_file(path, levels[li].data.data(),
                               levels[li].data.size() * sizeof(float));
                }
            if (!hfre_json.empty()) {
                vlx::HfreConfig hc2;
                hc2.roi_output_size = fc.roi_output_size;
                hc2.roi_aligned = fc.roi_aligned;
                vlx::HfreEncoder hfre(hc2);
                std::vector<vlx::Box> boxes;
                std::string json_txt;
                {
                    std::ifstream jf(hfre_json);
                    std::string line;
                    while (std::getline(jf, line)) json_txt += line;
                }
                size_t pos2 = 0;
                while ((pos2 = json_txt.find('[', pos2)) != std::string::npos) {
                    size_t end = json_txt.find(']', pos2);
                    if (end == std::string::npos) break;
                    float v[4]; int nb = 0;
                    size_t p2 = pos2;
                    while (nb < 4 && p2 < end) {
                        char* pe = nullptr;
                        v[nb] = std::strtof(json_txt.c_str() + p2, &pe);
                        if (pe == json_txt.c_str() + p2) { ++p2; continue; }
                        ++nb;
                        p2 = size_t(pe - json_txt.c_str());
                    }
                    if (nb == 4) boxes.push_back({v[0], v[1], v[2], v[3]});
                    pos2 = end + 1;
                }
                const float strides[4] = {4.f, 8.f, 16.f, 32.f};
                std::vector<float> vt;
                std::string verr;
                if (!hfre.encode_vt(boxes, levels,
                                    std::vector<float>(strides, strides + 4),
                                    &vt, &verr)) {
                    std::fprintf(stderr, "error: %s\n", verr.c_str());
                    return 1;
                }
                std::printf("hfre vt ok: %zu boxes x %zu dims\n", boxes.size(),
                            boxes.empty() ? 0 : vt.size() / boxes.size());
                if (!dump_vt.empty() && !vt.empty())
                    write_file(dump_vt, vt.data(), vt.size() * sizeof(float));
            }
        }
        std::printf("mmproj ok: ll %zu floats, final %zu floats\n",
                    ll_tokens.size(), final_tokens.size());
        ggml_backend_free(be);
        return 0;
    }
    if (cmd == "objfeat") {
        // objfeat --radio radio.gguf --mmproj mm.gguf --fpn fpn.gguf
        //        --input-img-bin hwc.bin --height H --width W
        //        --patches patches.bin --grid-h gh --grid-w gw
        //        --hfre boxes.json [--dump-fpn P] [--dump-aux o] [--dump-obj o]
        // Full aux+vt HFRE chain: RADIO levels (aux) + pooler->SimpleFPN
        // levels (vt) -> fuse (concat + sine pos embed) -> mlp2x_gelu
        // projector -> (N, 4096) <objfeat> token embeddings.
        std::string radio_m, mm_m, fpn_m, img_bin, patches_bin, hfre_json;
        std::string dump_fpn, dump_aux, dump_obj, dump_vt, dump_fused;
        int H = 0, W = 0, gh = 0, gw = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
            if (a == "--radio") radio_m = next();
            else if (a == "--mmproj") mm_m = next();
            else if (a == "--fpn") fpn_m = next();
            else if (a == "--input-img-bin") img_bin = next();
            else if (a == "--height") H = std::atoi(next());
            else if (a == "--width") W = std::atoi(next());
            else if (a == "--patches") patches_bin = next();
            else if (a == "--grid-h") gh = std::atoi(next());
            else if (a == "--grid-w") gw = std::atoi(next());
            else if (a == "--hfre") hfre_json = next();
            else if (a == "--dump-fpn") dump_fpn = next();
            else if (a == "--dump-aux") dump_aux = next();
            else if (a == "--dump-obj") dump_obj = next();
            else if (a == "--dump-vt") dump_vt = next();
            else if (a == "--dump-fused") dump_fused = next();
            else { std::fprintf(stderr, "unknown objfeat arg: %s\n", a.c_str()); return 1; }
        }
        if (radio_m.empty() || mm_m.empty() || fpn_m.empty() || img_bin.empty() ||
            !H || !W || patches_bin.empty() || gh <= 0 || gw <= 0 || hfre_json.empty()) {
            std::fprintf(stderr, "objfeat: --radio --mmproj --fpn --input-img-bin --height "
                                 "--width --patches --grid-h --grid-w --hfre required\n");
            return 1;
        }
        ggml_backend_t be = init_cpu_backend();
        if (!be) { std::fprintf(stderr, "no CPU backend\n"); return 1; }

        // ---- aux tower (RADIO): image -> 4 levels -> RoIAlign -> (N, 4608) ----
        vlx::GgufFile rguf, mguf, fguf;
        std::string gerr;
        if (!rguf.load(radio_m, &gerr) || !mguf.load(mm_m, &gerr) ||
            !fguf.load(fpn_m, &gerr)) {
            std::fprintf(stderr, "gguf load failed: %s\n", gerr.c_str());
            return 1;
        }
        vlx::RadioConfig rc;
        rc.patch_size = int(rguf.i32("vlx.radio.patch_size", rc.patch_size));
        rc.embed_dim = int(rguf.i32("vlx.radio.embed_dim", rc.embed_dim));
        rc.depth = int(rguf.i32("vlx.radio.depth", rc.depth));
        rc.num_heads = int(rguf.i32("vlx.radio.num_heads", rc.num_heads));
        rc.num_cls_tokens = int(rguf.i32("vlx.radio.num_cls_tokens", rc.num_cls_tokens));
        rc.num_registers = int(rguf.i32("vlx.radio.num_registers", rc.num_registers));
        for (int i = 0; i < 4; ++i)
            rc.out_indices[i] = i == 3 ? rc.depth - 1
                                       : int(std::lround(float(i + 1) * rc.depth / 4.f)) - 1;
        vlx::HfreConfig hc;
        hc.aux_spatial_scale = float(rguf.f32v("vlx.hfre.aux_spatial_scale", 1.0 / rc.patch_size));
        hc.roi_output_size = int(fguf.i32("vlx.hfre.roi_output_size", 14));
        hc.roi_aligned = fguf.i32("vlx.hfre.roi_aligned", 1) != 0;

        std::vector<uint8_t> raw_img, raw_patches;
        if (!read_file(img_bin, &raw_img) ||
            raw_img.size() != size_t(H) * W * 3 * 4) {
            std::fprintf(stderr, "error: --input-img-bin must be f32 HWC %dx%d\n", H, W);
            return 1;
        }
        std::vector<vlx::FeatureLevel> aux_levels;
        vlx::RadioGraph radio(be, rc, &rguf);
        if (!radio.run(reinterpret_cast<const float*>(raw_img.data()), H, W,
                       &aux_levels, &gerr)) {
            std::fprintf(stderr, "error: %s\n", gerr.c_str());
            return 1;
        }

        // ---- main tower (mmproj) -> pooler -> SimpleFPN -> (N, 2048) vt ----
        vlx::MmprojConfig mc;
        mc.patch_size = int(mguf.i32("vlx.mmproj.patch_size", mc.patch_size));
        mc.temporal = int(mguf.i32("vlx.mmproj.temporal", mc.temporal));
        mc.embed_dim = int(mguf.i32("vlx.mmproj.embed_dim", mc.embed_dim));
        mc.depth = int(mguf.i32("vlx.mmproj.depth", mc.depth));
        mc.num_heads = int(mguf.i32("vlx.mmproj.num_heads", mc.num_heads));
        mc.mlp_hidden = int(mguf.i32("vlx.mmproj.mlp_hidden", mc.mlp_hidden));
        mc.hidden_out = int(mguf.i32("vlx.mmproj.hidden_out", mc.hidden_out));
        mc.merge = int(mguf.i32("vlx.mmproj.merge", mc.merge));
        mc.pos_side = int(mguf.i32("vlx.mmproj.pos_side", mc.pos_side));
        mc.rope_theta = float(mguf.f32v("vlx.mmproj.rope_theta", mc.rope_theta));
        const size_t Np = size_t(gh) * gw;
        if (!read_file(patches_bin, &raw_patches) ||
            raw_patches.size() != Np * mc.kdim() * 4) {
            std::fprintf(stderr, "error: --patches must be %zu f32 values\n", Np * mc.kdim());
            return 1;
        }
        vlx::MmprojGraph mm(be, mc, &mguf);
        std::vector<float> ll_tokens, final_tokens;
        if (!mm.run(reinterpret_cast<const float*>(raw_patches.data()), gh, gw,
                    &ll_tokens, &final_tokens)) {
            std::fprintf(stderr, "mmproj run failed\n");
            return 1;
        }
        vlx::SimpleFpnConfig fc;
        fc.vt_dim = int(fguf.i32("vlx.fpn.vt_dim", mc.hidden_out));
        fc.out_channels = int(fguf.i32("vlx.fpn.out_channels", 512));
        fc.roi_output_size = hc.roi_output_size;
        fc.roi_aligned = hc.roi_aligned;
        vlx::SimpleFpnGraph fpn(be, fc, &fguf);
        std::vector<vlx::FeatureLevel> vt_levels;
        if (!fpn.run(ll_tokens.data(), gh, gw, &vt_levels)) {
            std::fprintf(stderr, "fpn run failed\n");
            return 1;
        }
        if (!dump_fpn.empty())
            for (size_t li = 0; li < vt_levels.size(); ++li) {
                char path[512];
                std::snprintf(path, sizeof(path), "%s_fpn_L%zu.bin", dump_fpn.c_str(), li);
                write_file(path, vt_levels[li].data.data(),
                           vt_levels[li].data.size() * sizeof(float));
            }

        // ---- boxes (original-image pixels), same parser as the other cmds ----
        std::vector<vlx::Box> boxes;
        std::string json_txt;
        {
            std::ifstream jf(hfre_json);
            std::string line;
            while (std::getline(jf, line)) json_txt += line;
        }
        size_t pos2 = 0;
        while ((pos2 = json_txt.find('[', pos2)) != std::string::npos) {
            size_t end = json_txt.find(']', pos2);
            if (end == std::string::npos) break;
            float v[4]; int nb = 0;
            size_t p2 = pos2;
            while (nb < 4 && p2 < end) {
                char* pe = nullptr;
                v[nb] = std::strtof(json_txt.c_str() + p2, &pe);
                if (pe == json_txt.c_str() + p2) { ++p2; continue; }
                ++nb;
                p2 = size_t(pe - json_txt.c_str());
            }
            if (nb == 4) boxes.push_back({v[0], v[1], v[2], v[3]});
            pos2 = end + 1;
        }

        vlx::HfreEncoder hfre(hc);
        std::vector<float> aux_nxc, vt_nxc;
        if (!hfre.encode_aux(boxes, aux_levels, &aux_nxc, &gerr)) {
            std::fprintf(stderr, "error: %s\n", gerr.c_str());
            return 1;
        }
        if (!dump_aux.empty() && !aux_nxc.empty())
            write_file(dump_aux, aux_nxc.data(), aux_nxc.size() * sizeof(float));
        const float strides[4] = {4.f, 8.f, 16.f, 32.f};
        if (!hfre.encode_vt(boxes, vt_levels,
                            std::vector<float>(strides, strides + 4), &vt_nxc, &gerr)) {
            std::fprintf(stderr, "error: %s\n", gerr.c_str());
            return 1;
        }
        if (!dump_vt.empty() && !vt_nxc.empty())
            write_file(dump_vt, vt_nxc.data(), vt_nxc.size() * sizeof(float));

        // ---- fuse: concat[aux(4608), vt(2048)] + sine pos embed ----
        // Pos-embed frame = the vision tower INPUT size in pixels:
        // merged-grid extent / vt_spatial_scale = (gw/2) * (16*2).
        std::vector<vlx::FeatureLevel> vt_grid_ref(1);
        vt_grid_ref[0].width = (gw / 2) * int(1.f / hc.vt_spatial_scale);
        vt_grid_ref[0].height = (gh / 2) * int(1.f / hc.vt_spatial_scale);
        std::vector<float> fused;
        if (!hfre.fuse(aux_nxc, vt_nxc, boxes, int(aux_nxc.size() / std::max<size_t>(boxes.size(), 1)),
                       int(vt_nxc.size() / std::max<size_t>(boxes.size(), 1)),
                       vt_grid_ref, &fused, &gerr)) {
            std::fprintf(stderr, "error: %s\n", gerr.c_str());
            return 1;
        }

        // ---- aux projector (mlp2x_gelu): Linear -> erf-GELU -> Linear ----
        // Mini single-output ggml graph over the (N, 6656) fused rows;
        // output (N, 4096) row-major = the <objfeat> token embeddings.
        if (!dump_fused.empty() && !fused.empty())
            write_file(dump_fused, fused.data(), fused.size() * sizeof(float));
        std::vector<float> objfeat;
        if (!boxes.empty() &&
            !vlx::run_mlp2x_gelu(be, fguf, "aux_proj", fused, int64_t(boxes.size()), &objfeat)) {
            std::fprintf(stderr, "error: aux_proj projector failed\n");
            return 1;
        }
        std::printf("objfeat ok: %zu boxes x %zu dims (LLM hidden)\n", boxes.size(),
                    boxes.empty() ? 0 : objfeat.size() / boxes.size());
        if (!dump_obj.empty() && !objfeat.empty())
            write_file(dump_obj, objfeat.data(), objfeat.size() * sizeof(float));
        ggml_backend_free(be);
        return 0;
    }
    if (cmd != "radio") { print_usage(); return cmd == "--help" ? 0 : 1; }

    std::string model, input_bin, dump_prefix, hfre_json, out_indices_str;
    int H = 0, W = 0, img_w = 0, img_h = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        if (a == "--model") model = next();
        else if (a == "--input-bin") input_bin = next();
        else if (a == "--height") H = std::atoi(next());
        else if (a == "--width") W = std::atoi(next());
        else if (a == "--dump") dump_prefix = next();
        else if (a == "--out-indices") out_indices_str = next();  // taps (debug)
        else if (a == "--hfre") hfre_json = next();
        else if (a == "--boxes-img-w") img_w = std::atoi(next());
        else if (a == "--boxes-img-h") img_h = std::atoi(next());
    }
    if (model.empty() || input_bin.empty() || H <= 0 || W <= 0) {
        print_usage();
        return 1;
    }

    vlx::GgufFile gguf;
    std::string err;
    if (!gguf.load(model, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    vlx::RadioConfig rc;
    rc.patch_size = int(gguf.i32("vlx.radio.patch_size", rc.patch_size));
    rc.embed_dim = int(gguf.i32("vlx.radio.embed_dim", rc.embed_dim));
    rc.depth = int(gguf.i32("vlx.radio.depth", rc.depth));
    rc.num_heads = int(gguf.i32("vlx.radio.num_heads", rc.num_heads));
    rc.mlp_hidden = int(gguf.i32("vlx.radio.mlp_hidden", rc.mlp_hidden));
    rc.num_cls_tokens = int(gguf.i32("vlx.radio.num_cls_tokens", rc.num_cls_tokens));
    rc.num_registers = int(gguf.i32("vlx.radio.num_registers", rc.num_registers));
    rc.pos_rows = int(gguf.i32("vlx.radio.pos_rows", rc.pos_rows));
    rc.pos_cols = int(gguf.i32("vlx.radio.pos_cols", rc.pos_cols));
    if (!out_indices_str.empty()) {
        // comma-separated block indices for the 4 taps (debug bisection)
        int n = 0;
        const char* oi = out_indices_str.c_str();
        char* endp = nullptr;
        while (n < 4 && *oi) {
            rc.out_indices[n++] = int(std::strtol(oi, &endp, 10));
            oi = endp && *endp == ',' ? endp + 1 : endp;
        }
    } else {
        // out_indices derived like _infer: round((i+1)*depth/4)-1, last = depth-1.
        for (int i = 0; i < 4; ++i)
            rc.out_indices[i] = i == 3 ? rc.depth - 1
                                       : int(std::lround(float(i + 1) * rc.depth / 4.f)) - 1;
    }

    vlx::HfreConfig hc;
    hc.aux_spatial_scale = float(gguf.f32v("vlx.hfre.aux_spatial_scale", 1.0 / rc.patch_size));
    hc.vt_spatial_scale = float(gguf.f32v("vlx.hfre.vt_spatial_scale", hc.vt_spatial_scale));
    hc.pos_embedding_dim = int(gguf.i32("vlx.hfre.pos_embedding_dim", hc.pos_embedding_dim));
    hc.roi_output_size = int(gguf.i32("vlx.hfre.roi_output_size", hc.roi_output_size));
    hc.roi_aligned = gguf.i32("vlx.hfre.roi_aligned", 0) != 0;
    hc.add_pos_embedding = gguf.i32("vlx.hfre.add_pos_embedding", 1) != 0;

    ggml_backend_t backend = init_cpu_backend();
    if (!backend) { std::fprintf(stderr, "error: cpu backend init failed\n"); return 1; }

    vlx::RadioGraph radio(backend, rc, &gguf);

    std::vector<uint8_t> raw;
    if (!read_file(input_bin, &raw) || raw.size() != size_t(H) * W * 3 * 4) {
        std::fprintf(stderr, "error: input.bin must be f32 HWC of %dx%d\n", H, W);
        return 1;
    }
    std::vector<vlx::FeatureLevel> levels;
    if (!radio.run(reinterpret_cast<const float*>(raw.data()), H, W, &levels, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::printf("radio ok: %zu levels %dx%d c%d\n", levels.size(),
                levels[0].height, levels[0].width, levels[0].channels);
    if (!dump_prefix.empty()) {
        for (size_t li = 0; li < levels.size(); ++li)
            dump_f32(dump_prefix, "radio_L" + std::to_string(li),
                     levels[li].data.data(),
                     levels[li].data.size() * sizeof(float));
    }

    // HFRE aux branch (no learned params): runs against the RADIO output and
    // is gated by scripts/test_roi_align.py against torchvision directly.
    if (!hfre_json.empty()) {
        vlx::HfreConfig hc2 = hc;
        vlx::HfreEncoder hfre(hc2);
        std::vector<float> aux_nxc;
        std::vector<vlx::Box> boxes;
        // boxes.json: list of [x1,y1,x2,y2] in ORIGINAL image pixels.
        std::string json_txt;
        {
            std::ifstream jf(hfre_json);
            std::string line;
            while (std::getline(jf, line)) json_txt += line;
        }
        // Minimal parse of a JSON array of 4-arrays (no JSON dep in CLI).
        size_t pos = 0;
        while ((pos = json_txt.find('[', pos)) != std::string::npos) {
            size_t end = json_txt.find(']', pos);
            if (end == std::string::npos) break;
            float v[4]; int n = 0;
            size_t p2 = pos;
            while (n < 4 && p2 < end) {
                char* pe = nullptr;
                v[n] = std::strtof(json_txt.c_str() + p2, &pe);
                if (pe == json_txt.c_str() + p2) { ++p2; continue; }
                ++n;
                p2 = size_t(pe - json_txt.c_str());
            }
            if (n == 4) boxes.push_back({v[0], v[1], v[2], v[3]});
            pos = end + 1;
        }
        if (!hfre.encode_aux(boxes, levels, &aux_nxc, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::printf("hfre aux ok: %zu boxes x %zu dims\n", boxes.size(),
                    boxes.empty() ? 0 : aux_nxc.size() / boxes.size());
        if (!dump_prefix.empty() && !aux_nxc.empty())
            dump_f32(dump_prefix, "hfre_aux", aux_nxc.data(),
                     aux_nxc.size() * sizeof(float));
    }
    ggml_backend_free(backend);
    return 0;
}
