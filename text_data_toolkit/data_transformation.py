"""
Data transformation module
 Transform text data for NLP analysis.
"""
import re
import difflib
import pandas as pd
from text_data_toolkit import data_cleaning as clean
from text_data_toolkit import eda as eda

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
    suffixes = ['ed', 'ing', 'ly', 's', 'es', 'tion', 'er']
    exceptions = {"this", "has", "his", "was", "thus", "gas", "class", 'during', 'better'}

    if exception_words is not None:
        exceptions.update(exception_words)

    word_list = tokenize_text(text)
    stemmed_words = []

    for word in word_list:
        if word.lower() in exceptions:
            stemmed_words.append(word)
            continue

        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2 :
                word = word[:-len(suffix)]
                break
        stemmed_words.append(word)

    stemmed_words_string = " ".join(stemmed_words)
    return stemmed_words_string

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

        matches = difflib.get_close_matches(stem, english_words, n=5, cutoff=0.75)
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
    stemmed = basic_stem_words(no_stop, exception_words = None)
    autocorrected = autocorrect_text(stemmed, exception_words = None)

    return autocorrected

def dataframe_all_transform(df, text_column, custom_stopword = None, exception_words = None, new_column = "Transformed Text"):
    """ Takes in a dataframe and text column and does all the data transformation steps
        Removes Stopwords, Stems Words, Autocorrects Words """
    df[new_column] = df[text_column].apply(
        lambda x: textdata_all_transform(x, custom_stopword = custom_stopword, exception_words = exception_words))

    return df

def label_data_sentiment(data, text_column = None, new_column = "Sentiment",
                         custom_positive = None, custom_negative = None,
                         negation_bigram = None):
    """Label text data into categories (sentiment analysis)"""
    positive_words = {
        "amazing", "love", "like", "good", "great", "awesome", "wonderful", "fantastic",
        "fabulous", "excellent", "outstanding", "brilliant", "superb", "delightful",
        "pleased", "enjoy", "charming", "cheerful", "happy", "satisfied", "nice", "cool",
        "impressive", "terrific", "marvelous", "splendid", "adorable", "beautiful", "best"}
    negative_words = {
        "bad", "terrible", "hate", "awful", "disgusting", "sad", "unpleasant", "horrible",
        "disappointing", "suck", "worst", "nasty", "gross", "angry", "depressing", "dreadful",
        "lame", "poor", "boring", "annoying", "mediocre", "painful", "unhappy", "regret",
        "frustrating", "cringe", "crappy", "pathetic"}
    negation_words = {
        "not", "never", "no", "don't", "didn't", "isn't", "wasn't", "won't", "can't",
        "doesn't", "hasn't", "hadn't", "couldn't", "wouldn't", "shouldn't", "ain't",
        "mightn't", "mustn't", "neither", "nor"}

    if custom_positive is not None:
        positive_words.update(set(custom_positive))

    if custom_negative is not None:
        negative_words.update(set(custom_negative))


    negation_bigrams = set(negation_bigram or [])

    if negation_bigrams:
        temp_negation_bigrams = set()
        for neg_word in negation_words:
            for bigram in negation_bigrams:
                temp_negation_bigrams.add((f"{neg_word}", bigram))

        negation_bigrams = temp_negation_bigrams
    else:
        negation_bigrams = set()

    def lexicon_score(text):
        if not isinstance(text, str):
            return "Neutral"

        tokens = tokenize_text(text)
        bigrams = eda.generate_ngrams(tokens, 2)
        trigrams = eda.generate_ngrams(tokens, 3)

        score = 0
        skip_index = set()

        # First Check for Trigrams
        for i, (first, second, third) in enumerate(trigrams):
            if first in negation_words and third in positive_words:
                score -= 1
                skip_index.update({i, i+1, i+2})

            elif first in negation_words and third in negative_words:
                score -= 1
                skip_index.update({i, i+1, i+2})

            elif f"{first} {second}" in negation_bigrams:
                score -= 1
                skip_index.update({i, i+1, i+2})
            elif third in negative_words:
                score -= 1
                skip_index.update({i+2})

        for i, (first, second) in enumerate(bigrams):
            if i in skip_index or (i+1) in skip_index:
                continue

            if first in negation_words and second in positive_words:
                score -= 1
                skip_index.update({i, i+1})

            elif first in negation_words and second in negative_words:
                score += 1
                skip_index.update({i, i+1})

        for i, t in enumerate(tokens):
            if i in skip_index:
                continue
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

    if isinstance(data, pd.DataFrame):
        data[new_column] = data[text_column].apply(lexicon_score)
        return data

    else:
        return lexicon_score(data)

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
    common_skills = {"python", "nlp", "javascript", "sql", "html", "cloud", "react", "snowflake", "pyspark", "tableau", "pytorch", "scikit", "regex", "spark", "machine learning"}

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