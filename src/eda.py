"""
Exploratory Data Analysis module
 Generate exploratory data analysis summaries.
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import src.data_transformation as dt

def generate_wordcloud(data, custom_stopwords=None):
    """Generate a word cloud from text data"""

    # handle stopwords
    base_stopwords = {'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', 'for', 'with', 'on',
        'at', 'from', 'by', 'up', 'about', 'into', 'over', 'after', 'under',
        'above', 'below', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'but', 'what', 'why',
        'can', 'could', 'should', 'would', 'how', 'when', 'where', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'i', 'he', 'she', 'it', 'they',
        'them', 'my', 'his', 'her', 'its', 'our', 'their', 'you', 'your', 'yours'}

    if custom_stopwords is not None:
        # homogenize stopwords and update list
        if isinstance(custom_stopwords, str):
            custom_stopwords = custom_stopwords.lower()
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
    else:
        raise TypeError("Data must be a string, list, or pandas series")

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
        raise ValueError("text_column cannot be None")
    else:
        if df[text_column].dtype != "object":
            raise TypeError("Text column must be of type 'object'")

    output_dict = {"document_stats": {}, "length_stats": {}, "word_stats": {}, "frequent_words": {}}

    # document stats
    output_dict["document_stats"]["total_docs"] = df[text_column].shape[0]
    output_dict["document_stats"]["empty_docs"] = df[text_column].isnull().sum()
    output_dict["document_stats"]["unique_docs"] = df[text_column].nunique()

    # length stats
    len_total = 0
    len_list = []
    for i in df[text_column]:
        len_total += len(i)
        len_list.append(len(i))

    mean = len_total / df[text_column].shape[0]

    len_list.sort()

    # find median
    if len(len_list) % 2 == 1: # odd length
        middle_index = len(len_list) // 2
        median = len_list[middle_index]
    else:
        middle_index_1 = len(len_list) // 2 - 1
        middle_index_2 = len(len_list) // 2
        median = (len_list[middle_index_1] + len_list[middle_index_2]) / 2

    output_dict["length_stats"]["min_length"] = min(len_list)
    output_dict["length_stats"]["max_length"] = max(len_list)
    output_dict["length_stats"]["char_count_mean"] = mean
    output_dict["length_stats"]["char_count_median"] = median

    # word stats
    word_counter = [] # how many words are in each record
    for i in df[text_column]:
        word_counter.append(len(i.split()))

    # total words
    word_count_sum = sum(word_counter)

    # unique words
    unique_words = set()
    for i in df[text_column]:
        for x in i.split():
            unique_words.add(x)

    # avg word length
    word_len_list = []
    for i in df[text_column]:
        for x in i.split():
            word_len_list.append(len(x))

    output_dict["word_stats"]["avg_words_per_doc"] = word_count_sum / df[text_column].shape[0]
    output_dict["word_stats"]["total_words"] = word_count_sum
    output_dict["word_stats"]["unique_words"] = len(unique_words)
    output_dict["word_stats"]["avg_word_length"] = sum(word_len_list) / len(word_len_list)

    # frequent words
    # handle stopwords
    base_stopwords = {'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', 'for', 'with', 'on',
        'at', 'from', 'by', 'up', 'about', 'into', 'over', 'after', 'under',
        'above', 'below', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'but', 'what', 'why',
        'can', 'could', 'should', 'would', 'how', 'when', 'where', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'i', 'he', 'she', 'it', 'they',
        'them', 'my', 'his', 'her', 'its', 'our', 'their', 'you', 'your', 'yours'}

    if custom_stopwords is not None:
        # homogenize stopwords and update list
        if isinstance(custom_stopwords, str):
            custom_stopwords = custom_stopwords.lower()
        else:
            for i in range(len(custom_stopwords)):
                custom_stopwords[i] = custom_stopwords[i].lower()

        base_stopwords.update(custom_stopwords)

    # get non-NULL records before exploding
    non_null_texts = df[text_column].dropna()

    words = non_null_texts.str.split().explode()
    word_counts = Counter(words) # count all words

    # remove stop words
    for key in base_stopwords:
        if key in word_counts:
            del word_counts[key]

    # get top 10 words
    top_10_counts = word_counts.most_common(10)

    output_dict["frequent_words"] = {word: count for word, count in top_10_counts}

    return output_dict

def plot_sentiment_distribution(df, text_column):
    """Visualize sentiment distribution"""
    if text_column is None:
        raise ValueError("text_column cannot be None")
    else:
        if df[text_column].dtype != "object":
            raise TypeError("Text column must be of type 'object'")

    df_plot = dt.label_data_sentiment(df, text_column)

    sns.catplot(data=df_plot, x="Sentiment", kind="count", hue="Sentiment", palette="viridis", height=7, aspect=1.5)

    plt.title("Sentiment Count Distribution")
    plt.show()


def top_ngrams(text_data, n=2, top_k=10):
    """Find most common n-grams in text"""
    pass
