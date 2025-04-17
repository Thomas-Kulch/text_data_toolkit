"""
Data transformation module
 Transform text data for NLP analysis.
"""
import re
import difflib
import pandas as pd
from text_data_toolkit import data_cleaning as clean
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


english_datafiles = ["../data/unigram_freq.csv", "../data/words_alpha.txt"]
dfs = clean.load_text_to_df(english_datafiles, line_length = 1)
df_unigrams, df_validwords = dfs['unigram_freq'], dfs['words_alpha']
df_validwords = df_validwords.rename(columns={"Column 0": "word" })

english_df = pd.merge(df_unigrams, df_validwords, on='word', how='inner')
english_df = clean.clean_dataframe_no_dups(english_df, "word")

contractions = {"don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "won't",
                "wouldn't", "couldn't", "can't", "i'm", "you're", "we're", "they're", "it's",
                "i've", "you've", "we've", "they've", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
                "there's", "that's", "what's", "who's", "where's", "how's", "let's", "hadn't", "shouldn't"}

positive_words = {
        "amazing", "love", "loved", "like", "liked", "good", "great", "awesome", "wonderful", "fantastic",
        "fabulous", "excellent", "outstanding", "brilliant", "superb", "delightful",
        "pleased", "enjoy", "charming", "cheerful", "happy", "satisfied", "nice", "cool",
        "impressive", "terrific", "marvelous", "splendid", "adorable", "beautiful", "best"}

negative_words = {
        "bad", "terrible", "hate", "awful", "disgusting", "sad", "unpleasant", "horrible",
        "disappointing", "suck", "worst", "nasty", "gross", "angry", "depressing", "dreadful",
        "lame", "poor", "boring", "annoying", "mediocre", "painful", "unhappy", "regret",
        "frustrating", "cringe", "crappy", "pathetic"}

def tokenize_text(text):
    """Split text into tokens (words)"""
    tokens = re.split(r"[^A-Za-z0-9']+", text.lower())
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
        base_stopwords.update(set(custom_stopword))

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
        for i in range(len(data)):
            data[i] = remove_singular_stopword(data[i])
        return data

def basic_stem_words(text, exception_words = None):
    """Stem words returns a string"""
    suffixes = ["tion", "ment", "ness", "ing", "ion", "ful", "ous", "ly", "ed", "es", "er", "s"]
    exceptions = {"this", "has", "his", "was", "thus", "gas", "class", 'during', 'better'}

    if exception_words is not None:
        exceptions.update(exception_words)

    word_list = tokenize_text(text)
    stemmed_words = []

    for w in word_list:
        original_word = w
        if w in exceptions:
            stemmed_words.append(w)
            continue

        for suffix in suffixes:
            if w.endswith(suffix) and len(w) > len(suffix) + 2 :
                new_word = w[:-len(suffix)]

                if new_word in english_df["word"].tolist():
                    w = new_word
                    break
                else:
                    w = new_word
                    break
        else:
            w = original_word

        stemmed_words.append(w)

    return " ".join(stemmed_words)

def autocorrect_text(text, exception_words = None):
    """Autocorrect words after rough stemming"""
    if exception_words is None:
        exception_words = set()

    english_words = english_df["word"].tolist()
    stemmed = basic_stem_words(text, exception_words = exception_words)

    original = tokenize_text(text.lower())
    stemmed = tokenize_text(stemmed.lower())

    words = text.lower().split()
    corrected = []

    for stem, original in zip(stemmed, original):
        if ((stem and original in exception_words)
            or stem in english_words\
            or stem in contractions\
            or stem == original):
            corrected.append(stem)
            continue

        matches = difflib.get_close_matches(stem, english_words, n=5, cutoff=0.75    )
        if matches:
            match_df = english_df[english_df["word"].isin(matches)]
            best_match = match_df.sort_values(by = "count", ascending = False).iloc[0]["word"]
            corrected.append(best_match)
        else:
            corrected.append(stem)

    corrected_string = " ".join(corrected)
    return corrected_string

def textdata_all_transform(text, text_column = None, custom_stopword = None, exception_words = None):
    """ Takes in text and does all the data transformation steps
        Removes Stopwords, Stems Words, Autocorrects Words """

    no_stop = remove_stopwords(text, text_column = text_column, custom_stopword = custom_stopword)
    autocorrected = autocorrect_text(no_stop, exception_words = exception_words)
    return autocorrected

def label_unique_total_job_skills(data, text_column = None, custom_skills = None):
    """Label text data into categories (job skills analysis)"""
    common_skills = {"python", "nlp", "java", "javascript", "sql", "html", "cloud", "react", "snowflake", "pyspark", "tableau", "pytorch", "scikit", "regex", "spark", "machine learning"}

    if custom_skills is not None:
        common_skills.update(custom_skills)

    skill_count_dict = {skill: 0 for skill in common_skills}

    def label_job_text(text):
        if not isinstance(text, str):
            return
        for skill in common_skills:
            if re.search(rf'\b{skill}\b', text):
                skill_count_dict[skill] += 1

    if isinstance(data, pd.DataFrame):
        for text in data[text_column]:
            label_job_text(text)

    else:
        label_job_text(data)

    filtered_dict = {}

    for skill, count in skill_count_dict.items():
        if count > 0:
            filtered_dict[skill] = count

    skill_count_dict = filtered_dict
    return skill_count_dict

def label_total_job_skills(data, text_column = None, custom_skills = None):
    """Label text data into categories (job skills analysis)"""
    common_skills = {"python", "nlp", "java", "javascript", "sql", "html", "cloud", "react", "snowflake", "pyspark", "tableau", "pytorch", "scikit", "regex", "spark", "machine learning"}

    if custom_skills is not None:
        common_skills.update(custom_skills)

    skill_count_dict = {skill: 0 for skill in common_skills}

    def label_job_text(text):
        if not isinstance(text, str):
            return
        for skill in common_skills:
            occurrence = len(re.findall(skill, text))
            skill_count_dict[skill] += occurrence

    if isinstance(data, pd.DataFrame):
        for text in data[text_column]:
            label_job_text(text)

    else:
        label_job_text(data)

    filtered_dict = {}

    for skill, count in skill_count_dict.items():
        if count > 0:
            filtered_dict[skill] = count

    skill_count_dict = filtered_dict
    return skill_count_dict

def split_data(df, target_column, train_size = 0.7, test_size = 0.15, random_state = 42):
    """Split data into train and test sets"""
    val_size = 1 - train_size - test_size
    if val_size < 0:
        raise ValueError("train_size + test_size must be less than 1")

    df_train_val, df_test = train_test_split(df, test_size = test_size,
                                             random_state = random_state, stratify = df[target_column])

    val_fraction = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(df_train_val, test_size = val_fraction,
                                        random_state = random_state, stratify = df_train_val[target_column])

    return df_train, df_val, df_test

def vectorize_text(series, method = "tfidf", max_features = 10000):
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features = max_features)
    elif method == "count":
        vectorizer = CountVectorizer(max_features = max_features)
    else:
        raise ValueError("Invalid method. Please choose from 'tfidf' or 'count'")

    vectorized_text = vectorizer.fit_transform(series)

    return vectorized_text, vectorizer