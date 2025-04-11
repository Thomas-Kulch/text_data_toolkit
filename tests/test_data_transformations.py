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
    expected_output = ["hello world", "i am lov this test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = data_transformation.basic_stem_words(test_text[i])
        assert returned == expected

def test_autocorrect_text():
    test_text = ["hello wrld", "i am lov thi test", "wordcloud test wrk"]
    expected_output = ["hello world", "i am love this test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = data_transformation.autocorrect_text(test_text[i], exception_words=["wordcloud"])
        assert returned == expected

def test_textdata_all_transform():
    test_text = ["hello ya wrld", "i am lov thi test", "wordcloud test wrk"]
    expected_output = ["hello world", "love this test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = data_transformation.textdata_all_transform(test_text[i],
                                                              custom_stopword = ['ya'],
                                                              exception_words=["wordcloud"])
        assert returned == expected

def test_dataframe_all_transform():
    test_data = {'text': ["hello ya wrld", "i am lov thi test", "wordcloud test wrk"]}
    test_df = pd.DataFrame(test_data)
    expected_output = ["hello world", "love this test", "wordcloud test work"]

    alltransform_df = data_transformation.dataframe_all_transform(test_df,
                                                                  text_column ="text",
                                                                  custom_stopword = ['ya'],
                                                                  exception_words=["wordcloud"])
    for i, expected in enumerate(expected_output):
        returned = alltransform_df.iloc[i]["Transformed Text"]
        assert returned == expected

def test_label_data_sentiment():
    pass

