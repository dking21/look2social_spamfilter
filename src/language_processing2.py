from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

regex = re.compile('[%s]' % re.escape(string.punctuation))
stopwords_ = stopwords.words('english')
punctuation_ = set(string.punctuation)

def filter_stopwords_punct(sentence):
    """remove stop words and punctuation"""
    sentence = regex.sub('', sentence)
    text = [word for word in sentence.split() if word not in stopwords_]
    return ' '.join(text)


from nltk.tokenize import word_tokenize

def sentence_to_words_clean(txt):
    """function to tokenzine sentence"""
    for sentence in txt:
        yield(word_tokenize(sentence))


from nltk.stem import WordNetLemmatizer

def lemming(text):
    """a function which stems each word in the given text"""
    lemmer = WordNetLemmatizer()
    text = [lemmer.lemmatize(word) for word in text]
    return text

# removing stopwords/punctuation --> tokenize --> lemmatize

def processing_text(target_df,column):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    stopwords_ = stopwords.words('english')
    punctuation_ = set(string.punctuation)
    target_df[column] = target_df[column].astype(str)
    data = target_df[column].values.tolist()
    data1 = list(filter_stopwords_punct(sentence) for sentence in data)
    data_words = list(sentence_to_words_clean(data1))
    data_lem = [lemming(sentence) for sentence in data_words]
    target_df['lemmed'] = data_lem
    target_df['lemmed'] = target_df['lemmed'].apply(lambda x: ', '.join(x))
    return target_df

def vectorize_text(target_df, max_feature_number):
    vectorizer = TfidfVectorizer(stop_words='english', max_features = max_feature_number)
    X = vectorizer.fit_transform(target_df['lemmed'])
    text_tfidf_matrix = X.toarray()
    col_name_lst = []
    for n in range(text_tfidf_matrix.shape[1]):
        col_name = f'text-TF-IDF-{n}'
        target_df[col_name] = text_tfidf_matrix[:, n]
        col_name_lst.append(col_name)
    return col_name_lst