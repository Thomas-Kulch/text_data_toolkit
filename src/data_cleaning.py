"""
Data cleaning module
 Clean and normalize text data.
"""
import os
import pandas as pd
import re
import fuzzywuzzy


def load_text_to_df(files, columns=None):
    """Load text files into a pandas DataFrame
    files - list of file paths
    columns - list of column names, default ["filename", "text"]
    df: pd.DataFrame, dataframe with each row containing a filename and text
    """
    all_dfs = []

    for file in files:
        filename = os.path.basename(file)
        extension = os.path.splitext(filename)[-1].lower()

        if extension == '.csv':
            df_temp = pd.read_csv(file, quoting = 3)

        elif extension == '.tsv':
            df_temp = pd.read_csv(file, sep="\t", quoting = 3)

        else:
           df_temp = pd.read_csv(file, quoting = 3)

    if columns is not None and len(columns) == len(df_temp.columns):
        df_temp.columns = columns

    all_dfs.append(df_temp)

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)

    return df

def handle_missing_values(df, text_column):
    """Handle missing values in text data"""
    df[text_column].replace('', pd.NA, inplace=True)
    df.dropna(subset=[text_column], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def remove_duplicates_fuzzy(df, text_column, threshold = 90):
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
            fuzzy_ratio = fuzzywuzzy.ratio(text_i, text_j)

            if fuzzy_ratio > threshold:
                drop.add(j)
    df.drop(drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def normalize_text(df, text_column):
    """Basic text normalization (lowercase, remove punctuation, etc.)"""

    # Convert to lowercase
    df[text_column] = df[text_column].str.lower()

    # Remove Punctuation
    df[text_column] = df[text_column].str.replace(r'[^\w\s]', '', regex=True)

    # Remove Whitespaces
    df[text_column] = df[text_column].str.replace(r'\s+', ' ', regex=True)

    return df

def clean_dataframe(df, text_column):
    """Apply all cleaning steps to a dataframe"""
    # Handle Missing Values
    df = handle_missing_values(df, text_column)

    # Remove_Duplicates
    df = remove_duplicates_fuzzy(df, text_column, threshold=90)

    # Normalize Text
    df = normalize_text(df, text_column)

    return df