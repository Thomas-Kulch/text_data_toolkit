"""Unit tests for data_transformation and label sentiment modules"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from text_data_toolkit import data_transformation as transform
from text_data_toolkit import data_cleaning as clean
from text_data_toolkit import label_sentiment as lss

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
    test_text = ["hello wonderful world", "i am loving this test", "this wordcloud test works"]
    expected_output = ["hello wonder world", "i am love this test", "this wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = transform.autocorrect_text(test_text[i], exception_words=["wordcloud"])
        assert returned == expected

def test_textdata_all_transform():
    test_text = ["hello ya world", "i am loving this test", "wordcloud test work"]
    expected_output = ["hello world", "love test", "wordcloud test work"]

    for i, expected in enumerate(expected_output):
        returned = transform.textdata_all_transform(test_text[i],
                                                    custom_stopword = ['ya'],
                                                    exception_words=["wordcloud"])
        assert returned == expected

def test_label_data_sentiment_str():
    test_text = ["test so great",
                 "amazingly terrible test",
                 "bad test"]
    test_text = [transform.basic_stem_words(i) for i in test_text]
    test_text = [transform.autocorrect_text(i) for i in test_text]

    expected_output = ["Positive", "Negative", "Negative"]

    for i, expected in enumerate(expected_output):
        returned = lss.label_data_sentiment(test_text[i])
        assert returned == expected

def test_label_data_sentiment_df():
    test_data = {'text' : ["test so great",
                 "amazingly terrible test",
                 "bad test"]}
    test_df = pd.DataFrame(test_data)

    test_df['text'] = test_df['text'].apply(transform.basic_stem_words)
    test_df['text'] = test_df['text'].apply(transform.autocorrect_text)
    test_df["Sentiment"] = test_df['text'].apply(lss.label_data_sentiment)
    expected_output = ["Positive", "Negative", "Negative"]

    for i, expected in enumerate(expected_output):
        returned = test_df.iloc[i]["Sentiment"]
        assert returned == expected

def test_sentiment_features():
    test_text = "test so bad yet great"
    expected_output = pd.Series([1, 1, 0])
    returned = lss.sentiment_features(test_text)

    assert returned.equals(expected_output)

def test_label_unique_job_skills_str():
    test_text = ["Python, SQL required",
                 "No requirements",
                 "python best must have python only python",
                 "Must have python, SQl , machine learning, java"]
    test_text = [clean.normalize_data(i) for i in test_text]

    expected_output = ({'python': 1, 'sql': 1}, {}, {'python': 1},
                       {'python': 1, 'sql': 1, 'machine learning': 1, 'java': 1})

    for i, expected in enumerate(expected_output):
        returned = transform.label_unique_total_job_skills(test_text[i])
        assert returned == expected

def test_label_unique_job_skills_df():
    test_data = {'text': ["Python, SQL required",
                          "No requirements",
                          "python best must have python only python",
                          "Must have python, SQl , machine learning, java"]}

    test_df = pd.DataFrame(test_data)
    test_df["text"] = clean.clean_dataframe_no_dups(test_df, 'text')
    expected_output = ({'python': 3, 'sql': 2, 'machine learning': 1, 'java': 1})
    returned = transform.label_unique_total_job_skills(test_df, text_column = "text")
    assert returned == expected_output

def test_label_total_job_skills_str():
    test_text = ["Python, SQL required",
                 "No requirements",
                 "python best must have python only python",
                 "Must have python, SQl , machine learning, java java"]
    test_text = [clean.normalize_data(i) for i in test_text]

    expected_output = ({'python': 1, 'sql': 1}, {}, {'python': 3},
                       {'python': 1, 'sql': 1, 'machine learning': 1, 'java': 2})

    for i, expected in enumerate(expected_output):
        returned = transform.label_total_job_skills(test_text[i])
        assert returned == expected

def test_label_total_job_skills_df():
    test_data = {'text': ["Python, SQL required",
                 "No requirements",
                 "python best must have python only python",
                 "Must have python, SQl , machine learning, java java"]}

    test_df = pd.DataFrame(test_data)
    test_df["text"] = clean.clean_dataframe_no_dups(test_df, 'text')
    expected_output = ({'python': 5, 'sql': 2, 'machine learning': 1, 'java': 2})
    returned = transform.label_total_job_skills(test_df, text_column = "text")
    assert returned == expected_output

def test_split_data():
    data = {'text': ['sample text'] * 100, 'label': [0, 1] * 50}
    df = pd.DataFrame(data)

    train, val, test = transform.split_data(df, target_column='label', train_size=0.7, test_size=0.15)
    total = len(df)

    assert abs(len(train) - total * 0.7) <= 1
    assert abs(len(val) - total * 0.15) <= 1
    assert abs(len(test) - total * 0.15) <= 1

def test_vectorize_text():
    series = pd.Series(["this is a test", "another test sentence"])
    X, vectorizer = transform.vectorize_text(series, method="tfidf")
    X2, vectorizer2 = transform.vectorize_text(series, method="count")

    assert X.shape[0] == 2
    assert X2.shape[0] == 2
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(vectorizer2, CountVectorizer)