"""Pinned original checkpoints required by the three-stage Kaggle experiment."""

TEACHER = {
    "repository": "apple/DFN5B-CLIP-ViT-H-14",
    "revision": "11738501a1db6d5e0a3451a71ba100be02e577e6",
    "filename": "open_clip_pytorch_model.bin",
    "sha256": "d67de50faa7f3ddce52fbab4f4656b04686a0bb15c26ebd0144d375cfa08b8ae",
}
STUDENT = {
    "filename": "ViT-B-32.pt",
    "sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}
