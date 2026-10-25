"""Original frozen teacher checkpoints and the OpenAI ViT-B/32 baseline."""

TEACHERS = {
    "dfn5b_h14": dict(
        model="ViT-H-14-quickgelu", pretrained="dfn5b", source_branch="main",
        repository="apple/DFN5B-CLIP-ViT-H-14",
        filename="open_clip_pytorch_model.bin",
        revision="11738501a1db6d5e0a3451a71ba100be02e577e6",
        sha256="d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae",
    ),
    "dfn2b_l14": dict(
        model="ViT-L-14", pretrained="dfn2b_s39b",
        source_branch="experiment/teacher-dfn2b-l14-s39b",
        repository="apple/DFN2B-CLIP-ViT-L-14-39B",
        filename="open_clip_pytorch_model.bin",
        revision="a3dd4900c25439e55266b7d5f94138e0cc0a80e0",
        sha256="f2f724646a73b4cd392e5bb41dae589e17ba52dab83006457dddca739d3e2cf9",
    ),
    "laion_h14": dict(
        model="ViT-H-14", pretrained="laion2b_s32b_b79k",
        source_branch="experiment/teacher-laion-h14",
        repository="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        filename="open_clip_model.safetensors",
        revision="1c2b8495b28150b8a4922ee1c8edee224c284c0c",
        sha256="0084e75319a50ad85ef45377bad5bc38f2f58824459eb690048d51c9f8863be5",
    ),
    "metaclip_l14": dict(
        model="ViT-L-14", pretrained="metaclip_fullcc",
        source_branch="experiment/teacher-metaclip-l14",
        repository="timm/vit_large_patch14_clip_224.metaclip_2pt5b",
        filename="open_clip_pytorch_model.bin",
        revision="1c09b97e09fb19bba2bf4548c9c7cca4b87f7888",
        sha256="8982bc381934bb39006c26e370d12da1f1eff5748762489f7b22e1a350975061",
    ),
    "openai_l14": dict(
        model="ViT-L-14", pretrained="openai",
        source_branch="experiment/teacher-openai-l14",
        repository="timm/vit_large_patch14_clip_224.openai",
        filename="open_clip_model.safetensors",
        revision="18d0535469bb561bf468d76c1d73aa35156c922b",
        sha256="9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e",
    ),
    "openai_b32": dict(
        model="ViT-B-32", pretrained="openai", source_branch="main (student backbone)",
        repository="openai/CLIP", filename="ViT-B-32.pt", checkpoint_format="torchscript",
        url="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        sha256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    ),
}

DATASET_ROOTS = {
    "sketchy_1": "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy",
    "sketchy_2": "/kaggle/input/datasets/b20dccn616nguynhutun/sketchy/Sketchy",
    "tuberlin": "/kaggle/input/datasets/b20dccn616nguynhutun/tuberlin",
    "quickdraw": "/kaggle/input/datasets/b20dccn616nguynhutun/quickdraw",
}


def class_text(name):
    # Match main's class-name spelling; preserve hyphens and spaces.
    return name.replace("_", " ")


def prompt_templates(modality, mode):
    if modality not in {"sketch", "photo"}:
        raise ValueError(modality)
    if mode == "modality":
        return ["a sketch of a {}." if modality == "sketch" else "a photo of a {}."]
    if mode == "photo":
        return ["a photo of a {}."]
    if mode == "ensemble":
        return ["a photo of a {}.", "a sketch of a {}.", "a drawing of a {}."]
    raise ValueError(mode)
