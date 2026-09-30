"""Fixed photo/sketch templates used identically by teacher and student."""

TEXT_PROMPT_PAIRS = {
    "baseline": {
        "photo": "a photo of a {class}.",
        "sketch": "a sketch of a {class}.",
    },
    "depicting": {
        "photo": "a photograph depicting a {class}.",
        "sketch": "a drawing depicting a {class}.",
    },
    "shows": {
        "photo": "this photograph shows a {class}.",
        "sketch": "this drawing shows a {class}.",
    },
    "line_drawing": {
        "photo": "a photograph of a {class}.",
        "sketch": "a line drawing of a {class}.",
    },
    "hand_drawn": {
        "photo": "an image showing a {class}.",
        "sketch": "a hand-drawn sketch of a {class}.",
    },
}


def prompt_pair_config(pair="baseline"):
    if pair not in TEXT_PROMPT_PAIRS:
        raise ValueError(f"Unknown text prompt pair: {pair}")
    return {"name": pair, **TEXT_PROMPT_PAIRS[pair]}


def class_texts(classnames, modality, pair="baseline"):
    if modality not in ("photo", "sketch"):
        raise ValueError(f"Unsupported text modality: {modality}")
    template = prompt_pair_config(pair)[modality]
    return [
        template.format_map({"class": name.replace("_", " ")})
        for name in classnames
    ]
