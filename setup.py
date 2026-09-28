from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tmrm",
    version="4.1.0",
    author="Balaji P, Navaneetham V, Dhavan RG",
    author_email="balajikrishnan031@gmail.com",
    description="Topological Manifold Resonant Machine: Unified Non-Parametric ML Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/balajikrishnan031/TMRM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest", "scikit-learn", "matplotlib"],
    },
)
