"""MambaFedEdge: Mamba-2 Based Federated Learning for Edge AI Battery Management."""
from setuptools import setup, find_packages

setup(
    name="mambafededge",
    version="0.1.0",
    description="Mamba-2 based federated learning framework for edge AI battery management systems",
    author="MambaFedEdge Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "flask>=3.0.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov", "black", "flake8"],
        "gpu": ["triton>=2.1.0"],
    },
)
