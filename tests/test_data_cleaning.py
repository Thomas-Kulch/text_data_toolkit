"""Unit tests for data_cleaning module"""
import os
import pytest
import tempfile
import pandas as pd
from text_data_toolkit import data_cleaning as clean

@pytest.fixture
def test_temporary_files():
    files = []

    # Temporary CSV
    csv_content = 'name,age\nAlice,24\nBob,19'
    csv_file = tempfile.NamedTemporaryFile(mode='w', suffix = '.csv', delete=False)
    csv_file.write(csv_content)
    csv_file.close()
    files.append(csv_file.name)

    # Temporary TSV
    tsv_content = 'name\tage\nAlice\t24\nBob\t19'
    tsv_file = tempfile.NamedTemporaryFile(mode='w', suffix = '.tsv', delete=False)
    tsv_file.write(tsv_content)
    tsv_file.close()
    files.append(tsv_file.name)

    # Temporary Any
    txt_content = 'name\nage\nAlice\n24\nBob\n19'
    txt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    txt_file.write(txt_content)
    txt_file.close()
    files.append(txt_file.name)

    yield files
    for file in files:
        os.remove(file)

def test_csv_load_text_to_df(test_temporary_files):
    csv_file = []

    for f in test_temporary_files:
        if f.endswith('.csv'):
            csv_file.append(f)

    dfs_dict = clean.load_text_to_df(csv_file)
    key = os.path.splitext(os.path.basename(csv_file[0]))[0]
    df = dfs_dict[key]

    assert df.shape == (2, 2)
    assert list(df.columns) == ['name', 'age']
    assert df.iloc[0, 0] == 'Alice'

def test_tsv_load_text_to_df(test_temporary_files):
    tsv_file = []

    for f in test_temporary_files:
        if f.endswith('.tsv'):
            tsv_file.append(f)

    dfs_dict = clean.load_text_to_df(tsv_file)
    key = os.path.splitext(os.path.basename(tsv_file[0]))[0]
    df = dfs_dict[key]

    assert df.shape == (2, 2)
    assert list(df.columns) == ['name', 'age']
    assert df.iloc[0, 0] == 'Alice'

def test_any_load_text_to_df(test_temporary_files):
    txt_file = []

    for f in test_temporary_files:
        if f.endswith('.txt'):
            txt_file.append(f)

    dfs_dict = clean.load_text_to_df(txt_file, columns = ['name', 'age'], line_length = 2)
    key = os.path.splitext(os.path.basename(txt_file[0]))[0]
    df = dfs_dict[key]

    assert df.shape == (3, 2)
    assert list(df.columns) == ['name', 'age']
    assert df.iloc[1, 0] == 'Alice'

def test_remove_duplicates_fuzzy():
    test_data = {'text':["Hello world", "This is a test", "WOrdcloud test", "Hello world"]}
    test_df = pd.DataFrame(test_data)
    expected_output = {'text':["Hello world", "This is a test", "WOrdcloud test"]}

    no_dups_df = clean.remove_duplicates_fuzzy(test_df, "text")
    for i, expected in enumerate(expected_output["text"]):
        returned = no_dups_df.iloc[i]["text"]
        assert returned == expected

def test_normalize_text():
    test_data = {'text': ["Hello world.?", "This     is    a test", "WOrdclOud teSt/^&"]}
    test_df = pd.DataFrame(test_data)
    expected_output = {'text': ["hello world", "this is a test", "wordcloud test"]}

    normalized_df = clean.normalize_text(test_df, "text")
    for i, expected in enumerate(expected_output["text"]):
        returned = normalized_df.iloc[i]["text"]
        assert returned == expected

def test_handle_missing_values():
    test_data = {'text': ["Hello world", "This is a test", "WOrdclOud teSt/^&", "     ", None]}
    test_df = pd.DataFrame(test_data)
    expected_output = {'text': ["Hello world", "This is a test", "WOrdclOud teSt/^&"]}

    no_missing_df = clean.handle_missing_values(test_df, "text")
    for i, expected in enumerate(expected_output["text"]):
        returned = no_missing_df.iloc[i]["text"]
        assert returned == expected

def test_clean_dataframe_no_dups():
    test_data = {'text': ["Hello world", "This is a test", "WOrdclOud teSt/^&", "     ", None]}
    test_df = pd.DataFrame(test_data)
    expected_output = {'text': ["hello world", "this is a test", "wordcloud test"]}

    cleaned_no_dups_df = clean.clean_dataframe_no_dups(test_df, "text")
    for i, expected in enumerate(expected_output["text"]):
        returned = cleaned_no_dups_df.iloc[i]["text"]
        assert returned == expected

def test_clean_dataframe():
    test_data = {'text': ["Hello world", "This is a test", "hello World", "hello wrld", "WOrdclOud teSt/^&", "     ", None]}
    test_df = pd.DataFrame(test_data)
    expected_output = {'text': ["hello world", "this is a test", "wordcloud test"]}

    cleaned_df = clean.clean_dataframe(test_df, "text")
    for i, expected in enumerate(expected_output["text"]):
        returned = cleaned_df.iloc[i]["text"]
        assert returned == expected


def main():
    test_remove_duplicates_fuzzy()
    test_normalize_text()
    test_handle_missing_values()
    test_clean_dataframe_no_dups()
    test_clean_dataframe()

if __name__ == "__main__":
    main()