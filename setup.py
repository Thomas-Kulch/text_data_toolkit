from setuptools import setup, find_packages
import os

# Read long description safely
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="text_data_toolkit",
    version="0.1.0",
    author="Thomas Kulch and Ben Lin",
    author_email="kulch.t@northeastern.edu",
    description="A toolkit for handling and transforming natural language data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Thomas-Kulch/text_data_toolkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.2.0",
        "wordcloud>=1.9.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "python-Levenshtein>=0.27.0",
        "nltk>=3.9.0",
        "fuzzywuzzy @ git+https://github.com/seatgeek/fuzzywuzzy.git@0.18.0"
    ],
    extras_require={
        "dev": ["pytest>=8.3.0"]
    },
    include_package_data=True,
)
