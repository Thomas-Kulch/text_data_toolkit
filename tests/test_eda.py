"""Unit tests for eda module"""
import pytest
import pandas as pd
from wordcloud import WordCloud
from text_data_toolkit import eda

# generate_wordcloud testing
def test_generate_wordcloud_with_list():
    data = ["Hello world", "This is a test", "WOrdcloud test"]
    wc = eda.generate_wordcloud(data)
    assert isinstance(wc, WordCloud)
    assert "hello" in wc.words_ # this comes from the WordCloud object
    assert "wordcloud" in wc.words_

def test_generate_wordcloud_with_series():
    data_list = ["Hello world", "This is a test", "WOrdcloud test"]
    data = pd.Series(data_list)
    wc = eda.generate_wordcloud(data)
    assert isinstance(wc, WordCloud)
    assert "world" in wc.words_

def test_generate_wordcloud_with_string_and_no_stopwords():
    data = "Hello world. This is a test. WOrdcloud TEST"
    wc = eda.generate_wordcloud(data)
    assert isinstance(wc, WordCloud)
    assert "a" not in wc.words_ # see if stopwords were excluded
    assert "test" in wc.words_

def test_generate_wordcloud_with_custom_stopwords():
    data = "Hello world. This is a test. WOrdcloud TEST"
    wc = eda.generate_wordcloud(data, "hello")
    assert isinstance(wc, WordCloud)
    assert "a" not in wc.words_
    assert "hello" not in wc.words_

def test_generate_wordcloud_invalid_type():
    with pytest.raises(TypeError):
        eda.generate_wordcloud(12345)  # invalid input

# dataframe with text column for testing
text_column = ["test", 'hello', 'world', 'green', 'a', 'up', 'hello', 'hello']
int_column = [4, 7, 3, 6, 8, 12, 1, 2]
df = pd.DataFrame({"text_column": text_column, "int_column": int_column})

# text_summary_stats testing
def test_text_summary_stats_invalid_inputs():
    with pytest.raises(TypeError):
        eda.text_summary_stats(df)

    with pytest.raises(TypeError):
        eda.text_summary_stats(df, "int_column")

def test_text_summary_stats_document_stats():
    output = eda.text_summary_stats(df, "text_column")
    assert isinstance(output, dict)
    assert "document_stats" in output.keys()
    assert output["document_stats"]["unique_docs"] > 0

def test_text_summary_stats_length_stats():
    output = eda.text_summary_stats(df, "text_column")
    assert output["length_stats"]["max_length"] == 5
    assert output["length_stats"]["min_length"] == 1
    assert output["length_stats"]["char_count_mean"] == 4
    assert output["length_stats"]["char_count_median"] == 5
    assert output["length_stats"]["total_length"] == 32

def test_text_summary_stats_word_stats():
    output = eda.text_summary_stats(df, "text_column")
    assert output["word_stats"]["unique_words"] == 6
    assert output["word_stats"]["avg_words_per_doc"] == 1
    assert output["word_stats"]["total_words"] == 8
    assert output["word_stats"]["avg_word_length"] == 4

def test_text_summary_stats_frequent_words_with_stopwords():
    output = eda.text_summary_stats(df, "text_column", ["world", "hello"])
    assert output["frequent_words"]["green"] == 1
    assert "hello" not in output["frequent_words"]

# plot_sentiment_distribution testing
def test_plot_sentiment_distribution_column():
    text_column = "text_column"
    output = eda.plot_sentiment_distribution(df, "text_column")
    assert f'{text_column}_sentiment' in output.columns

# top_ngrams testing
def test_top_ngrams_series():
    with pytest.raises(TypeError):
        eda.top_ngrams(df["int_column"])

    count = eda.top_ngrams(df["text_column"], n=2, top_k=4)
    assert (("test", "hello"), 1) in count


def test_top_ngrams_string():
    count = eda.top_ngrams("hello world this is a test of what this is testing", n=2, top_k=4)
    assert (("this", "is"), 2) in count

def test_top_ngrams_stopwords():
    count = eda.top_ngrams("hello world this is a test of what this is testing", "this", n=2, top_k=4)
    assert (("this", "is"), 2) not in count

def test_top_ngrams_stopwords_list():
    count = eda.top_ngrams("hello world this is a test of what this is testing", ["this", "is"], n=2, top_k=4)
    assert (("this", "is"), 2) not in count

