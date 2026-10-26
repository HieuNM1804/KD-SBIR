"""Shared non-CUDA dependencies for Colab and the Kaggle offline bundle."""
RUNTIME_ROOTS = [
    "open-clip-torch==3.2.0", "transformers==4.57.6", "huggingface-hub<1.0",
    "safetensors", "ftfy", "regex", "numpy", "pillow", "tqdm", "packaging",
    "pytorch-lightning>=2.5,<3", "torchmetrics>=1.6,<2", "lightning-utilities", "tensorboard",
]
