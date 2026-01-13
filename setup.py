from setuptools import setup, find_packages

setup(
    name="sz_diffusion",
    version="0.1.0",
    description="Diffusion Models for SZ Map Generation",
    author="Luca Fontana",
    packages=find_packages(),
    install_requires=[
        "torch",
        "ema-pytorch",
        "torchvision",
        "numpy",
        "tqdm",
        "matplotlib",
        "einops",
        "pandas",
        # "tensorboard",    # Uncomment if tensorboard is needed. Currently used only for logging the training process,
                            # which is not included for this setup.
    ],
    python_requires=">=3.8",
)