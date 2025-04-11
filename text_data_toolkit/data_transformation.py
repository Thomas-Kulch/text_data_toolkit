"""
Data transformation module
 Transform text data for NLP analysis.
"""
import re
import difflib
import pandas as pd

def tokenize_text(text):
    """Split text into tokens (words)"""
    tokens = re.split(r'[^A-Za-z0-9]+', text.lower())
    # Remove empty strings
    for i in tokens:
        if i == '':
            tokens.remove(i)

    return tokens

def tokenize_dataframe(df, column, new_column = "Tokenized Text"):
    df[new_column] = df[column].apply(tokenize_text).str.join(', ')
    return df

def remove_stopwords(data, text_column, custom_stopword = None, new_column = "Removed Stopwords"):
    """Remove common stopwords"""
    base_stopwords = {
        'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', 'for', 'with', 'on',
        'at', 'from', 'by', 'up', 'about', 'into', 'over', 'after', 'under',
        'above', 'below', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'but', 'what', 'why',
        'can', 'could', 'should', 'would', 'how', 'when', 'where', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'i', 'he', 'she', 'it', 'they',
        'them', 'my', 'his', 'her', 'its', 'our', 'their', 'you', 'your', 'yours',
        "we", "so"}

    if custom_stopword is not None:
        base_stopwords.update(custom_stopword)

    def remove_singular_stopword(text):
        tokens = tokenize_text(text)
        filtered_tokens = []
        for t in tokens:
            if t not in base_stopwords:
                filtered_tokens.append(t)
        filtered_tokens_string = " ".join(filtered_tokens)
        return filtered_tokens_string

    if isinstance(data, str):
        return remove_singular_stopword(data)

    elif isinstance(data, pd.DataFrame):
        data[new_column] = data[text_column].apply(remove_singular_stopword)
        return data

    else:
        raise TypeError("Data must be a string or a pandas dataframe")

def basic_stem_words(text):
    """Stem words returns a string"""
    suffixes = ['ed', 'ing', 'ly', 's', 'es']
    word_list = tokenize_text(text)
    stemmed_words, og_words = [], []

    for word in word_list:
        og_word = word
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1 :
                word = word[:-len(suffix)]
                break
        stemmed_words.append(word)
        og_words.append(og_word)

    stemmed_words_string = " ".join(stemmed_words)

    return stemmed_words_string

def autocorrect_stem_words(text, cutoff = 0.85):
    """Autocorrect words after rough stemming"""
    with open("words_alpha.txt") as f:
        english_words = f.read().splitlines()

    words = basic_stem_words(text).lower().split()
    corrected = []

    for w in words:
        matches = difflib.get_close_matches(w, english_words, n=1, cutoff=cutoff)
        if matches:
            corrected.append(matches[0])
        else:
            corrected.append(w)

    corrected_string = " ".join(corrected)
    return corrected_string


def textdata_all_transform(text, custom_stopword = None, cutoff = 0.8):
    """ Takes in text and does all the data transformation steps
        Removes Stopwords, Stems Words, Autocorrects Words """

    no_stop = remove_stopwords(text, custom_stopword = custom_stopword)
    stemmed = basic_stem_words(no_stop)
    autocorrected = autocorrect_text(stemmed, cutoff = cutoff)

    return autocorrected

def dataframe_all_transform(df, text_column, custom_stopword = None, cutoff = 0.8, new_column = "Transformed Text"):
    """ Takes in a dataframe and text column and does all the data transformation steps
        Removes Stopwords, Stems Words, Autocorrects Words """
    df[new_column] = df[text_column].apply(
        lambda x: textdata_all_transform(x, custom_stopword = custom_stopword, cutoff = cutoff))

    return df

def label_data_sentiment(df, text_column, new_column = "Sentiment"):
    """Label text data into categories (sentiment analysis)"""
    positive_words = {"amazing", "love", "like", "good", "great", "awesome", "amazingly"}
    negative_words = {"bad", "terrible", "hate", "awful", "disgusting", "sad", "unpleasant", "horrible", "disappointing"}

    def lexicon_score(text):
        tokens = tokenize_text(text)
        score = 0
        for t in tokens:
            if t in positive_words:
                score += 1
            elif t in negative_words:
                score -= 1

        if score > 0:
            return "Positive"
        if score < 0:
            return "Negative"
        else:
            return "Neutral"

    df[new_column] = df[text_column].apply(lexicon_score)
    return df

def label_job_skills(df, text_column, custom_skills = None):
    """Label text data into categories (job skills analysis)"""
    common_skills = {"python", "nlp", "javascript", "sql", "html", "cloud", "react", "snowflake", "pyspark", "tableau", "pytorch", "scikit", "regex", "spark", "machine learning"}
    if custom_skills is not None:
        common_skills.update(custom_skills)

    skill_count_dict = {skill: 0 for skill in common_skills}

    for text in df[text_column]:
        if not isinstance(text, str):
            continue
        for skill in common_skills:
            occurrence = len(re.findall(skill, text))
            skill_count_dict[skill] += occurrence

    return skill_count_dict