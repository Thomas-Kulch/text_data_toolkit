# Project documentation

**text_data_toolkit** is a Python package designed to assist in handling and transforming natural language data. It provides a suite of tools for efficient text processing, analysis, and visualization.

## Features

- **Data Cleaning**: Functions to preprocess and clean text data, including removal of stopwords, punctuation, and special characters.
- **Text Transformation**: Utilities for tokenization, stemming, lemmatization, and vectorization.
- **Visualization**: Tools to generate word clouds and other graphical representations of text data.
- **File Operations**: Functions to move, bulk rename, delete, and list files

## Installation

To install the package, clone the repository and use pip:

```bash
git clone https://github.com/Thomas-Kulch/text_data_toolkit.git
cd text_data_toolkit
pip install .
```

## Dependencies
#### The package requires the following Python libraries:

pandas>=2.2.0

wordcloud>=1.9.0

matplotlib>=3.10.0

seaborn>=0.13.0

python-Levenshtein>=0.27.0

nltk>=3.9.0

These dependencies will be installed automatically when you install the package.

## Usage

Here's a basic example of how to use the toolkit:

```python
import text_data_toolkit as tdt

# Sample text data
text = "This is a sample sentence for text processing."

# Clean the text
cleaned_text = tdt.clean_text(text)

# Tokenize the text
tokens = tdt.tokenize_text(cleaned_text)

# Generate a word cloud
tdt.generate_wordcloud(tokens)
```

## Testing
To run tests, use pytest from the root of the project:
```bash
pytest
```

Make sure you have all development dependencies installed. You can install them with:
```bash
pip install -e .[dev]
```

## Authors
**Thomas Kulch** - kulch.t@northeastern.edu

**Ben Lin** - lin.benj@northeastern.edu

## License
This project is licensed under the MIT License.