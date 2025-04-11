"""Unit tests for data_transformation module"""
import os
import pytest
import tempfile
import pandas as pd
from text_data_toolkit import data_transformation


def test_tokenize_text():
    test_text = ["Hello world", "This is a test", "WOrdcloud test"]
    expected_output = [["hello", "world"], ["this", "is", "a", "test"], ["wordcloud", "test"]]

    for i, expected in enumerate(expected_output):
        returned = data_transformation.tokenize_text(test_text[i])
        assert returned == expected

def test_tokenize_dataframe():
    test_data = {'text': ["Hello world", "This is a test", "Wordcloud test"]}
    test_df = pd.DataFrame(test_data)
    expected_output = ["hello, world", "this, is, a, test", "wordcloud, test"]

    tokenized_df = data_transformation.tokenize_dataframe(test_df, "text")

    for i, expected in enumerate(expected_output):
        returned = tokenized_df.iloc[i]["Tokenized Text"]
        assert returned == expected

def test_remove_stopwords():
    test_data = {'text': ["Hello the world", "This is a test", "Wordcloud test"]}
    test_df = pd.DataFrame(test_data)
    expected_output = ["hello world", "test", "wordcloud test"]

    stopwords_df = data_transformation.remove_stopwords(test_df, "text")
    for i, expected in enumerate(expected_output):
        returned = stopwords_df.iloc[i]["Removed Stopwords"]
        assert returned == expected

def test_basic_stem_words():
    test_text = ["Hello world", "I am loving this test", "Wordcloud test worked"]

    expected_output = ["hello world", "i am lov thi test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = data_transformation.basic_stem_words(test_text[i])
        assert returned == expected

def test_autocorrect_text():
    test_text = ["hello world", "i am lov thi test", "wordcloud test wrk"]
    expected_output = ["hello world", "i am love this test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = data_transformation.autocorrect_stem_words(test_text[i])
        assert returned == expected


def test_label_data_sentiment():
    pass


def main():
    test_tokenize_text()
    test_tokenize_dataframe()
    test_remove_stopwords()
    test_basic_stem_words()


if __name__ == "__main__":
    main()


