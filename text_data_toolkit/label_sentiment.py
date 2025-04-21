"""
Label Sentiment module
"""
from text_data_toolkit import eda as eda
from text_data_toolkit import data_transformation as dt
import json
import os
import pandas as pd

def label_data_sentiment(data, custom_positive = None, custom_negative = None,
                         filename = None, return_counts = False):
    """ Label text data into sentiment categories using a basic lexicon-based approach
    :param data: (str) input text to be labeled
    :param custom_positive: (list) Optional list of custom positive words
    :param custom_positive: (list) Optional list of custom negative words
    :param filename: (str) Optional path to a json file with additional positive/negative words
    :param return_counts: (bool) Whether to return counts (pos, neg, score) or just the label)
    :return: Sentiment label or return_counts (pos, neg, score))
    """
    negation_words = {
        "not", "never", "no", "don't", "didn't", "isn't", "wasn't", "won't", "can't",
        "doesn't", "hasn't", "hadn't", "couldn't", "wouldn't", "shouldn't", "ain't",
        "mightn't", "mustn't", "neither", "nor", "like"}

    pos_lex = dt.positive_words.copy()
    neg_lex = dt.negative_words.copy()

    if filename:
        if os.path.isfile(filename):
            with open(filename, 'r', encoding="utf-8") as f:
                existing_words = json.load(f)

            load_pos = set(existing_words.get("positive", []))
            load_neg = set(existing_words.get("negative", []))
            pos_lex.update(set(load_pos))
            neg_lex.update(set(load_neg))
    else:
        pass

    if custom_positive is not None:
        pos_lex.update(set(custom_positive))

    if custom_negative is not None:
        neg_lex.update(set(custom_negative))

    def lexicon_score(text):
        if not isinstance(text, str):
            return "Neutral, 0 , 0, 0"

        tokens = dt.tokenize_text(text)
        bigrams = eda.generate_ngrams(tokens, 2)

        pos_count = 0
        neg_count = 0
        skip_index = set()

        for i, (first, second) in enumerate(bigrams):
            if i in skip_index or (i+1) in skip_index:
                continue

            if first in negation_words and second in pos_lex:
                neg_count += 1
                skip_index.update({i, i+1})

            elif first in negation_words and second in neg_lex:
                pos_count += 1
                skip_index.update({i, i+1})

        for i, t in enumerate(tokens):
            if i in skip_index:
                continue
            if t in pos_lex:
                pos_count += 1
            elif t in neg_lex:
                neg_count += 1

        score = pos_count - neg_count

        if score > 0:
            return ("Positive", pos_count, neg_count, score)
        if score < 0:
            return ("Negative", pos_count, neg_count, score)
        else:
            return ("Neutral", pos_count, neg_count, score)

    if filename:
        updated_data = {"positive": list(pos_lex), "negative": list(neg_lex)}

        with open(filename, 'w', encoding="utf-8") as f:
            json.dump(updated_data, f, indent=4, ensure_ascii=False)

    label, pos_count, neg_count, score = lexicon_score(data)

    if return_counts == False:
        return label

    else:
        return (pos_count, neg_count, score)

def sentiment_features(text, filename = None):
    """ Generate the positive count, negative count, and difference score for text to be used in ML models.
    :param text: (str) input text to be labeled
    :param filename: Optional path to a json file with additional positive/negative words
    :return: Series of sentiment features (pos, neg, score)
    """
    pos_count, neg_count, score = label_data_sentiment(text, filename = filename, return_counts = True)

    return pd.Series([pos_count, neg_count, score])