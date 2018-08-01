import pandas as pd
from io import StringIO
import re

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize, sent_tokenize

#import stemmer
from nltk.stem.snowball import SnowballStemmer
default_stemmer = SnowballStemmer("english")


"""The Lemmatizer copied from sklearn's documentation
http://scikit-learn.org/stable/modules/feature_extraction.html
I used Snowball Stemmer instead of WordNetLemmatizer"""
class LemmaTokenizer(object):
    def __init__(self):
        self.sbs = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.sbs.stem(t) for t in word_tokenize(doc)]



def remove_stop_words():
    '''
    Remove any stopwards as needed
    '''
    default_stopwords = set( stopwords.words('english') ).union( set(ENGLISH_STOP_WORDS) )
    my_additional_stop_words = ('effectively', 'us', 'see', 'linkedin', 'may', 'lead','ax',
                            'must', 'lack', 'implication','aspx ', 'udrwdfeucsfx', 'docidredir', 'aspx', 'may', 'lead','ax',
                            'must', 'lack', 'implication', 'effective', 'aware', 'windowtext',
                           'style', 'border', 'font', '_layouts', 'wopiframe', 'fff', 'currentcolor',
                           'currentcolor', 'solid') # Add any additional stopwords we dont want and update the list
    default_stopwords = default_stopwords.union(my_additional_stop_words)
    return(default_stopwords)    

def clean_text(text,):
    """
    This function receives text and returns clean word-list
    """
    
    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]
    
    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])
    
    
    #Convert to lower case , so that Hi and hi are the same
    text = text.lower()
    text = ''.join(c for c in text if not c.isdigit())
    #text = stem_text(text) #stemming
    #text = remove_stopwords(text) #remove stopwords
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' ')
    
    return(text)