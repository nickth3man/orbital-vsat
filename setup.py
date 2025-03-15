"""
Setup script for the Voice Separation & Analysis Tool (VSAT) package.
"""

from setuptools import setup, find_packages

setup(
    name="vsat",
    version="0.1.0",
    author="VSAT Team",
    author_email="vsat@example.com",
    description="Voice Separation & Analysis Tool for processing and analyzing audio recordings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/vsat",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.10",
    install_requires=[
        line.strip() for line in open("requirements.txt") if not line.startswith("#")
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
    },
) 