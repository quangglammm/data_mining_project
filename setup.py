"""Setup script for the project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rice-yield-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Rice Yield Prediction System for Mekong Delta",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rice-yield-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rice-yield-predict=src.presentation.cli.main:main",
        ],
    },
)

