"""Benchmark cases shared by the C++/python runners (single source of truth).

Text prompts follow the official showcase style: REFERRING PHRASES that pick
specific objects out of a same-class set (fine-grained language grounding),
matching assets/seek1.5-*.png ("water bottles that are not on the cardboard
box", "person riding electric bike", ...). Images and what each phrase tests:

  demo_image.jpg      the official fruit photo — plain multi-class detection
                      ("orange; apple", 7 boxes), the multi-box capability
  demo_image2.jpg     12-person group photo — referring phrase picks ONE man
                      (yellow shirt) out of 12 same-class people
  000000039769.jpg    two tabby cats on a pink couch (COCO val2017) — referring
                      phrase "the cat closest to the camera" picks the right cat
                      (NOTE: the left cat has no clean full-body WeDetect
                      proposal — bare "cat" only retrieves the right cat; the
                      detector recall, not the LLM, is the ceiling there)
  000000397133.jpg    period kitchen — multi-class with a person (kept as the
                      documented edge-of-argmax divergence case)
  kitchen_modern.jpg  modern open kitchen (Wikimedia Commons) — spatial relation
                      ("the microwave oven above the kitchen counter")
  street_traffic.jpg  Manhattan taxi jam — open-vocabulary phrase ("the white
                      police van") picks one specific vehicle among dozens
  crosswalk_pedestrians.jpg  aerial crosswalk — action phrase ("the person
                      riding a bicycle")
  ebike_riders.jpg    Shenzhen intersection — the OFFICIAL showcase phrase
                      verbatim ("person riding electric bike")
  water_bottles_table.jpg  six bottles on papers — spatial relation ("the water
                      bottle standing next to the white plastic bag")
"""

CASES = [  # (image filename, task, text, lang) — identical prompts for both backends
    ("demo_image.jpg", "detection", "orange; apple", "en"),
    ("demo_image2.jpg", "detection", "the man wearing a yellow shirt", "en"),
    ("demo_image2.jpg", "grounding_single", "the person in a white shirt", "en"),
    ("000000039769.jpg", "detection", "the cat closest to the camera", "en"),
    ("000000397133.jpg", "detection", "bowl; spoon; person", "en"),
    ("kitchen_modern.jpg", "detection", "the microwave oven above the kitchen counter", "en"),
    ("street_traffic.jpg", "detection", "the white police van", "en"),
    ("crosswalk_pedestrians.jpg", "detection", "the person riding a bicycle", "en"),
    ("ebike_riders.jpg", "detection", "person riding electric bike", "en"),
    ("water_bottles_table.jpg", "detection", "the water bottle standing next to the white plastic bag", "en"),
]

BACKENDS = ("cpp_f16", "cpp_q8_0", "python")


def case_key(image_name: str, task: str) -> str:
    from pathlib import Path
    return f"{Path(image_name).stem}_{task}"
