"""
Data cleaning module
 Clean and normalize text data.
"""
import os
import pandas as pd
import re
from fuzzywuzzy import fuzz
import numpy as np


def load_text_to_df(files, columns=None, line_length = 1):
    """Load text files into a pandas DataFrame
    files - list of file paths
    columns - list of column names, default ["filename", "text"]
    df: pd.DataFrame, dataframe with each row containing a filename and text
    """
    dfs_dict = {}

    for file in files:
        filename = os.path.basename(file)
        key_name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[-1].lower()

        if extension == '.csv':
            df_temp = pd.read_csv(file, sep=',', quotechar ='"')
        elif extension == '.tsv':
            df_temp = pd.read_csv(file, sep="\t")
        else:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()

            if len(lines) % line_length != 0:
                print("Warning - Number of lines in file is not divisible by line_length. Truncating file.")

            records = []
            i = 0
            while i + line_length <= len(lines):
                record = []
                for j in range(line_length):
                    record.append(lines[i + j].strip())
                records.append(record)
                i += line_length
            if columns is None:
                columns = [f"Column {i}" for i in range(line_length)]
            elif len(columns) != line_length:
                raise ValueError("Column Length must equal line length")

            df_temp = pd.DataFrame(records, columns=columns)

            for i in df_temp.columns:
                df_temp[i] = df_temp[i].str.replace('\uFEFF', '', regex=False)

        if columns is not None and len(columns) == len(df_temp.columns):
            df_temp.columns = columns

        dfs_dict[key_name] = df_temp

    return dfs_dict

def homogenize_columns(df):
    df.columns = df.columns.str.lower().str.replace(r'\s', '_', regex=True)
    return df

def remove_duplicates_fuzzy(df, text_column = None, threshold = 90):
    """Detect and remove duplicate texts
    df - pandas dataframe
    text_column - name of the column containing text data
    threshold = threshold for fuzzy matching, higher = stricter

    Uses a nested loop where each row i is compared with every row j, (i+1, i+2...)
    if the fuzzy ratio is above the threshold, the row is dropped.
    """
    drop = set()

    for i in range(len(df)):
        if i in drop:
            continue
        text_i = df.loc[i, text_column]

        for j in range(i + 1, len(df)):
            if j in drop:
                continue
            text_j = df.loc[j, text_column]
            fuzzy_ratio = fuzz.ratio(text_i, text_j)

            if fuzzy_ratio > threshold:
                drop.add(j)
    df.drop(drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def normalize_data(data, text_column = None):
    """Basic text normalization (lowercase, remove punctuation, etc.)"""

    def normalize_text(text):
        text = text.lower()
        text = re.sub(r"[^\w\s']", '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    if isinstance(data, str):
        return normalize_text(data)

    else:
        data[text_column] = data[text_column].astype(str).apply(normalize_text)
        return data

def handle_missing_values(df, text_column = None):
    """Handle missing values in text data (NaN, empty strings)
    Run after text data is normalized to remove whitespace"""

    df[text_column] = df[text_column].replace(r'^\s*$', np.nan, regex=True)
    df.dropna(subset=[text_column], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def clean_dataframe_no_dups(df, text_column = None):
    """Apply all cleaning steps to a dataframe without duplicates"""
    # Homogenize Columns
    df = homogenize_columns(df)

    # Normalize Text
    df = normalize_data(df, text_column)

    # Handle Missing Values
    df = handle_missing_values(df, text_column)

    return df

def clean_dataframe(df, text_column, threshold = 90):
    """Apply all cleaning steps to a dataframe"""
    # Homogenize Columns
    df = homogenize_columns(df)

    # Normalize Text
    df = normalize_data(df, text_column)

    # Remove_Duplicates
    df = remove_duplicates_fuzzy(df, text_column, threshold = threshold)

    # Handle Missing Values
    df = handle_missing_values(df, text_column)

    return df