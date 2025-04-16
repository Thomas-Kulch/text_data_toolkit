"""
Exploratory Data Analysis module
 Generate exploratory data analysis summaries.
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from text_data_toolkit import data_transformation as dt
from text_data_toolkit import data_cleaning as dc
import re
from nltk.util import ngrams

STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', 'for', 'with', 'on',
        'at', 'from', 'by', 'up', 'about', 'into', 'over', 'after', 'under',
        'above', 'below', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'but', 'what', 'why',
        'can', 'could', 'should', 'would', 'how', 'when', 'where', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'i', 'he', 'she', 'it', 'they',
        'them', 'my', 'his', 'her', 'its', 'our', 'their', 'you', 'your', 'yours',
        "we", "so", "as"}

def generate_wordcloud(data, custom_stopwords=None):
    """Generate a word cloud from text data"""

    # handle stopwords
    base_stopwords = STOPWORDS

    if custom_stopwords is not None:
        # homogenize stopwords and update list
        if isinstance(custom_stopwords, str): # check if string
            custom_stopwords = [custom_stopwords.lower()]
        else:
            for i in range(len(custom_stopwords)):
                custom_stopwords[i] = custom_stopwords[i].lower()

        base_stopwords.update(custom_stopwords)

    # handle case where data is a list of strings or a pd series
    if isinstance(data, pd.Series):
        final_text_data = " ".join(data.dropna())
    elif isinstance(data, list):
        final_text_data = " ".join(data)
    elif isinstance(data, str):
        final_text_data = data
    else: # incorrect input dtype
        raise TypeError("Data must be a string, list, or pandas series")

    # homogenize string and get rid of punctuation
    final_text_data = re.sub(r'[^\w\s]', '', final_text_data.lower())


    # create word cloud object and return it
    word_cloud = WordCloud(width=800, height=400, background_color="white", max_words=100, stopwords=base_stopwords)
    word_cloud.generate_from_text(final_text_data.lower())
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    return word_cloud

def text_summary_stats(df, text_column, custom_stopwords=None):
    """Generate basic statistics for text data"""
    if text_column is None:
        raise TypeError("text_column cannot be None")
    else:
        if df[text_column].dtype != "object":
            raise TypeError("Text column must be of type 'object'")

    output_dict = {"document_stats": {}, "length_stats": {}, "word_stats": {}, "frequent_words": {}}

    # normalize data using method from data_cleaning
    df_copy = dc.normalize_data(df, text_column)

    # document stats
    output_dict["document_stats"]["total_docs"] = df_copy[text_column].shape[0]
    output_dict["document_stats"]["empty_docs"] = int(df_copy[text_column].isnull().sum())
    output_dict["document_stats"]["unique_docs"] = df_copy[text_column].nunique()

    # length stats
    len_total = 0
    len_list = []
    for i in df_copy[text_column]:
        len_total += len(i)
        len_list.append(len(i))

    mean = len_total / df_copy[text_column].shape[0]

    len_list.sort() # sort the list

    # find median
    if len(len_list) % 2 == 1: # list is odd length
        middle_index = len(len_list) // 2
        median = len_list[middle_index]
    else:
        middle_index_1 = len(len_list) // 2 - 1
        middle_index_2 = len(len_list) // 2
        median = (len_list[middle_index_1] + len_list[middle_index_2]) / 2

    # calculate the length stats
    output_dict["length_stats"]["min_length"] = min(len_list)
    output_dict["length_stats"]["max_length"] = max(len_list)
    output_dict["length_stats"]["total_length"] = len_total
    output_dict["length_stats"]["char_count_mean"] = mean
    output_dict["length_stats"]["char_count_median"] = median

    # word stats
    word_counter = [] # how many words are in each record
    for i in df_copy[text_column]:
        word_counter.append(len(i.split()))

    # total words
    word_count_sum = sum(word_counter)

    # unique words
    unique_words = set()
    for i in df_copy[text_column]:
        for x in i.split():
            unique_words.add(x)

    # avg word length
    word_len_list = []
    for i in df_copy[text_column]:
        for x in i.split():
            word_len_list.append(len(x))

    # calculate the word stats
    output_dict["word_stats"]["avg_words_per_doc"] = word_count_sum / df_copy[text_column].shape[0]
    output_dict["word_stats"]["total_words"] = word_count_sum
    output_dict["word_stats"]["unique_words"] = len(unique_words)
    output_dict["word_stats"]["avg_word_length"] = sum(word_len_list) / len(word_len_list)

    # frequent words
    # handle stopwords
    base_stopwords = STOPWORDS

    if custom_stopwords is not None:
        # homogenize stopwords and update list
        if isinstance(custom_stopwords, str):
            custom_stopwords = [custom_stopwords.lower()]
        else:
            for i in range(len(custom_stopwords)):
                custom_stopwords[i] = custom_stopwords[i].lower()

        base_stopwords.update(custom_stopwords)

    # get non-NULL records before exploding
    non_null_texts = df_copy[text_column].dropna()

    words = non_null_texts.str.split().explode()
    word_counts = Counter(words) # count all words

    # remove stop words
    for key in base_stopwords:
        if key in word_counts:
            del word_counts[key]

    # get top 10 words
    top_10_counts = word_counts.most_common(10)

    # output frequent words to stats dict
    output_dict["frequent_words"] = {word: count for word, count in top_10_counts}

    return output_dict

def plot_sentiment_distribution(df, text_column):
    """Visualize sentiment distribution"""
    if text_column is None:
        raise ValueError("text_column cannot be None")
    else:
        if df[text_column].dtype != "object":
            raise TypeError("Text column must be of type 'object'")

    # normalize data using method from data_cleaning
    df = dc.normalize_data(df, text_column)

    # label sentiments using method from data_cleaning
    df[f"{text_column}_sentiment"] = df[f"{text_column}"].apply(lambda x: dt.label_data_sentiment(x))

    # plot results
    s = sns.catplot(data=df, x=f"{text_column}_sentiment", kind="count", hue=f"{text_column}_sentiment", palette="viridis", height=7, aspect=1.5)
    plt.title("Sentiment Count Distribution", fontsize = 20)

    plt.show()

def generate_ngrams(tokens, n = 2):
    result = []
    for i in range(n):
        result.append(tokens[i:])
    zipped = list(zip(*result))
    return zipped

def top_ngrams(data, stopwords=None, n=2, top_k=10):
    """Extract and count the most frequent word combinations (n-grams) from text data to
    identify common phrases and collocations."""

    if stopwords is not None:
        # homogenize stopwords and update list
        if isinstance(stopwords, str):
            stopwords = [stopwords.lower()]

        stopwords_set = set(stopwords)

    # handle case where data is a list of strings or a pd series
    if isinstance(data, pd.Series):
        text_data = " ".join(data.dropna())
    elif isinstance(data, list):
        text_data = " ".join(data)
    elif isinstance(data, str):
        text_data = data
    else:
        raise TypeError("Data must be a string, list, or pandas series")

    # homogenize string and get rid of punctuation
    text_data = re.sub(r'[^\w\s]', '', text_data.lower())


    # tokenize text using method from data_transformation
    tokenized_final_text_data = dt.tokenize_text(text_data)

    # remove stopwords if necessary
    if stopwords:
        filtered_text_data = [word for word in tokenized_final_text_data if word not in stopwords_set]
    else:
        filtered_text_data = tokenized_final_text_data


    # generate n-grams from tokens
    ngrams_list = list(ngrams(filtered_text_data, n=n))

    if not ngrams_list:
        return []

    # generate counts
    ngram_counts = Counter(ngrams_list)

    # get top k ngrams
    top_ngram_counts = ngram_counts.most_common(top_k)

    return top_ngram_counts

