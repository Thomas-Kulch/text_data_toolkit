"""Unit tests for data_transformation module"""
import pandas as pd
from text_data_toolkit import data_transformation as transform
from text_data_toolkit import data_cleaning as clean

def test_tokenize_text():
    test_text = ["Hello world", "This is a test", "WOrdcloud test"]
    expected_output = [["hello", "world"], ["this", "is", "a", "test"], ["wordcloud", "test"]]

    for i, expected in enumerate(expected_output):
        returned = transform.tokenize_text(test_text[i])
        assert returned == expected

def test_tokenize_dataframe():
    test_data = {'text': ["Hello world", "This is a test", "Wordcloud test"]}
    test_df = pd.DataFrame(test_data)
    expected_output = ["hello, world", "this, is, a, test", "wordcloud, test"]

    tokenized_df = transform.tokenize_dataframe(test_df, "text")

    for i, expected in enumerate(expected_output):
        returned = tokenized_df.iloc[i]["Tokenized Text"]
        assert returned == expected

def test_remove_stopwords():
    test_data = {'text': ["Hello the world", "This is a test", "Wordcloud test"]}
    test_df = pd.DataFrame(test_data)
    expected_output = ["hello world", "test", "wordcloud test"]

    stopwords_df = transform.remove_stopwords(test_df, "text")
    for i, expected in enumerate(expected_output):
        returned = stopwords_df.iloc[i]["Removed Stopwords"]
        assert returned == expected

def test_basic_stem_words():
    test_text = ["Hello world", "I am loving this test", "Wordcloud test worked"]
    expected_output = ["hello world", "i am lov this test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = transform.basic_stem_words(test_text[i])
        assert returned == expected

def test_autocorrect_text():
    test_text = ["hello wrld", "i am lov thi test", "wordcloud test wrk"]
    expected_output = ["hello world", "i am love this test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = transform.autocorrect_text(test_text[i], exception_words=["wordcloud"])
        assert returned == expected

def test_textdata_all_transform():
    test_text = ["hello ya wrld", "i am lov thi test", "wordcloud test wrk"]
    expected_output = ["hello world", "love this test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = transform.textdata_all_transform(test_text[i],
                                                    custom_stopword = ['ya'],
                                                    exception_words=["wordcloud"])
        assert returned == expected

def test_dataframe_all_transform():
    test_data = {'text': ["hello ya wrld", "i am lov thi test", "wordcloud test wrk"]}
    test_df = pd.DataFrame(test_data)
    expected_output = ["hello world", "love this test", "wordcloud test work"]

    alltransform_df = transform.dataframe_all_transform(test_df,
                                                        text_column ="text",
                                                        custom_stopword = ['ya'],
                                                        exception_words=["wordcloud"])
    for i, expected in enumerate(expected_output):
        returned = alltransform_df.iloc[i]["Transformed Text"]
        assert returned == expected

def test_label_data_sentiment_str():
    test_text = ["test so great",
                 "amazingly terrible test",
                 "bad test",
                 "amazingly, terrible, wonderfully made test"]
    test_text = [transform.basic_stem_words(i) for i in test_text]
    test_text = [transform.autocorrect_text(i) for i in test_text]

    expected_output = ["Positive", "Neutral", "Negative", "Positive"]

    for i, expected in enumerate(expected_output):
        returned = transform.label_data_sentiment(test_text[i])
        assert returned == expected

def test_label_data_sentiment_df():
    test_data = {'text' : ["test so great",
                 "amazingly terrible test",
                 "bad test",
                 "amazingly, terrible, wonderfully made test"]}
    test_df = pd.DataFrame(test_data)

    test_df['text'] = test_df['text'].apply(transform.basic_stem_words)
    test_df['text'] = test_df['text'].apply(transform.autocorrect_text)
    test_df["Sentiment"] = test_df['text'].apply(transform.label_data_sentiment)
    expected_output = ["Positive", "Neutral", "Negative", "Positive"]

    for i, expected in enumerate(expected_output):
        returned = test_df.iloc[i]["Sentiment"]
        assert returned == expected

def test_label_job_skills_str():
    test_text = ["Python, SQL required",
                 "No requirements",
                 "python best must have python only python",
                 "Must have python, SQl , machine learning, java"]
    test_text = [clean.normalize_data(i) for i in test_text]

    expected_output = ({'python': 1, 'sql': 1}, {}, {'python': 1},
                       {'python': 1, 'sql': 1, 'machine learning': 1, 'java': 1})

    for i, expected in enumerate(expected_output):
        returned = transform.label_job_skills(test_text[i])
        assert returned == expected

def test_label_job_skills_df():
    test_data = {'text': ["Python, SQL required",
                          "No requirements",
                          "python best must have python only python",
                          "Must have python, SQl , machine learning, java"]}

    test_df = pd.DataFrame(test_data)
    test_df["text"] = clean.clean_dataframe_no_dups(test_df, 'text')
    expected_output = ({'python': 3, 'sql': 2, 'machine learning': 1, 'java': 1})
    returned = transform.label_job_skills(test_df, text_column = "text")
    assert returned == expected_output