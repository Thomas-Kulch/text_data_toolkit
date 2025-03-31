"""
Data transformation module
 Transform text data for NLP analysis.
"""
import re
from src.data_cleaning import clean_dataframe


def tokenize_text(text):
    """Split text into tokens (words)"""

    tokens = re.split(r'[^A-Za-z0-9]+', text.lower())
    # Remove empty strings
    for i in tokens:
        if i == '':
            tokens.remove(i)

    return tokens

def remove_stopwords(df, text_column, custom_stopword):
    """Remove common stopwords"""
    base_stopwords = {
        'a', 'an', 'the', 'and', 'or', 'in', 'of', 'to', 'for', 'with', 'on',
        'at', 'from', 'by', 'up', 'about', 'into', 'over', 'after', 'under',
        'above', 'below', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'but', 'what', 'why',
        'can', 'could', 'should', 'would', 'how', 'when', 'where', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'i', 'he', 'she', 'it', 'they',
        'them', 'my', 'his', 'her', 'its', 'our', 'their', 'you', 'your', 'yours'}

    if custom_stopword is not None:
        base_stopwords.update(custom_stopword)

    df[text_column] = clean_dataframe(df, text_column)

    def remove_singular_stopword(text):
        tokens = tokenize_text(text)
        for t in tokens:
            if t in base_stopwords:
                tokens.remove(t)

    df[text_column] = df[text_column].apply(remove_singular_stopword)

def label_data_sentiment(df, text_column, method='lexicon'):
    """Label text data into categories (sentiment analysis)"""
    positive_words = {"amazing", "love", "like", "good", "great", "awesome", "amazingly"}
    negative_words = {"bad", "terrible", "hate", "awful", "disgusting", "sad", "unpleasant"}

    def lexicon_score(text):
        tokens = tokenize_text(text)
        pos_score = 0
        neg_score = 0
        for t in tokens:
            if t in positive_words:
                pos_score += 1
            elif t in negative_words:
                neg_score -= 1

        if pos_score > neg_score:
            return "Positive"
        if pos_score < neg_score:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment"] = df[text_column].apply(lexicon_score)
    return df

def label_job_skills(job_description_text):
    """Label text data into categories (job skills analysis)"""
    common_skills = {"python", "java", "c++", "c#", "javascript", "sql", "html", "css", "react", "angular", "react-native"}
    for skill in common_skills:
        skill_count_dict = {skill: 0}

    # Cleans text data
    job_description_text_clean = clean_dataframe(job_description_text)

    for text in job_description_text:
        for skill in common_skills:
            occurences = len(re.findall(skill, text))
            skill_count_dict[skill] += occurences

    # Convert to Dataframe
    skill_df = pd.DataFrame.from_dict(skill_count_dict, orient='index')


    # Sort by Descending
    skill_df.sort_values(by=0, ascending=False, inplace=True)
    skill.df.reset_index(drop=True, inplace=True)

    return skill_df

def split_data(df, train_size=0.7, val_size=0.15, test_size=0.15):
    """Split data into training, validation, and testing sets"""
    pass

def vectorize_text(text_series, method='tfidf'):
    """Convert text to numerical vectors"""
    pass
