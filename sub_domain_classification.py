import pandas as pd
from io import StringIO
import re

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords

# https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda
def clean_text(text):
    """
    This function receives text and returns clean word-list
    """

    #Convert to lower case , so that Hi and hi are the same
    text = text.lower()
    text = ''.join(c for c in text if not c.isdigit())
    #text = stem_text(text) #stemming
    #text = remove_stopwords(text) #remove stopwords
    text = re.sub(r"what's", "what is", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    
    return(text)

def remove_stop_words():
    '''
    Remove any stopwards as needed
    '''
    default_stopwords = set( stopwords.words('english') ).union( set(ENGLISH_STOP_WORDS) )
    my_additional_stop_words = ('effectively', 'us', 'see', 'linkedin', 'may', 'lead','ax',
                            'must', 'lack', 'implication','aspx ', 'udrwdfeucsfx', 'docidredir', 'aspx', 'may', 'lead','ax',
                            'must', 'lack', 'implication', 'effective', 'aware', 'windowtext',
                           'style', 'border', 'font', '_layouts', 'wopiframe', 'fff', 'currentcolor',
                           'currentcolor', 'solid',
                                'ap','skype','implications','following','furthermore','xbox','live','implemented',
                                'high','wdg','opg','march', 'fy',
                               'ia',) # Add any additional stopwords we dont want and update the list
    default_stopwords = default_stopwords.union(my_additional_stop_words)
    return(default_stopwords)

df = pd.read_csv("all_it_audit_issues.csv")
df['IssueDesc'] = df['IssueDesc'].map(lambda com : clean_text(com))
df = df[df.IssueDesc != 'acp no details provided'] # remove any ACP that we identified based on Issue Name

