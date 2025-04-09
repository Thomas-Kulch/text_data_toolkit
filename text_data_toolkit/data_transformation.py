"""
Data transformation module
 Transform text data for NLP analysis.
"""
import re
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

def remove_stopwords(df, text_column, custom_stopword = None, new_column = "Removed Stopwords"):
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
        return filtered_tokens

    df[new_column] = df[text_column].apply(remove_singular_stopword).str.join(', ')
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

""" Not sure if necessary 
    # Convert to Dataframe
    skill_df = pd.DataFrame.from_dict(skill_count_dict, orient='index', columns = ['new_column'])
    print(skill_count_dict)
    # Sort by Descending
    skill_df.reset_index(inplace=True)
    skill_df.rename(columns={'index', 'skill'}, inplace=True)
    skill_df.sort_values(by='new_column', ascending=False, inplace=True)
    skill_df.reset_index(drop=True, inplace=True)"""



def split_data(df, train_size=0.7, val_size=0.15, test_size=0.15):
    """Split data into training, validation, and testing sets"""
    pass

def vectorize_text(text_series, method='tfidf'):
    """Convert text to numerical vectors"""
    pass
