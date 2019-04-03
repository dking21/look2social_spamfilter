import pandas as pd
import numpy as np
import unicodedata
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def join_sent_ngrams(input_tokens, n):
    # first add the 1-gram tokens
    ret_list = list(input_tokens)
    #then for each n
    for i in range(2,n+1):
        # add each n-grams to the list
        ret_list.extend(['-'.join(tgram) for tgram in ngrams(input_tokens, i)])

    return(ret_list)

def text_to_tfidf(df):
    '''
    1. remove accent from text
    2. tokenize word
    3. stem word
    4. ngram word
    5. TFIDF vectorize word, return it as a matrix.
    '''
    df2 = df
    df2['text_without_accent'] = df2['text'].apply(remove_accents)
    df2['text_word_tokens'] = df2['text_without_accent'].apply(word_tokenize)
    df2['processed_text'] = df2['text_without_accent'].apply(text_process)
    stemmer_porter = PorterStemmer()
    df2['stemmed'] = df2['processed_text'].apply(lambda x: list(map(stemmer_porter.stem, x)))
    df2['ngrammed'] = list(map(lambda x : join_sent_ngrams(x, 3), df2['stemmed']))
    bow_transformer = TfidfVectorizer(max_features=1000,analyzer=text_process).fit(df2['ngrammed'])
    messages_bow = bow_transformer.transform(df2['ngrammed'])
    tfidf_matrix = messages_bow.todense()
    return tfidf_matrix

def desc_to_tfidf(df):
    '''
    1. remove accent from text
    2. tokenize word
    3. stem word
    4. ngram word
    5. TFIDF vectorize word, return it as a matrix.
    '''
    df2 = df
    df2['text_without_accent'] = df2['description'].apply(remove_accents)
    df2['text_word_tokens'] = df2['text_without_accent'].apply(word_tokenize)
    df2['processed_text'] = df2['text_without_accent'].apply(text_process)
    stemmer_porter = PorterStemmer()
    df2['stemmed'] = df2['processed_text'].apply(lambda x: list(map(stemmer_porter.stem, x)))
    df2['ngrammed'] = list(map(lambda x : join_sent_ngrams(x, 3), df2['stemmed']))
    bow_transformer = TfidfVectorizer(max_features=1000,analyzer=text_process).fit(df2['ngrammed'])
    messages_bow = bow_transformer.transform(df2['ngrammed'])
    tfidf_matrix = messages_bow.todense()
    return tfidf_matrix
