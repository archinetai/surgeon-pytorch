from setuptools import find_packages, setup

setup(
    name="surgeon-pytorch",
    packages=find_packages(exclude=[]),
    version="0.0.4",
    license="MIT",
    description="Surgeon - PyTorch",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/surgeon-pytorch",
    keywords=[
        "artificial intelligence",
        "deep learning",
    ],
    install_requires=["torch>=1.6", "data-science-types>=0.2"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
