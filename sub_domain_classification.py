import pandas as pd
from io import StringIO
import re

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords

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

df2 = pd.read_csv("all_it_audit_issues.csv")
col2 = ['MSPP_Domain', 'MSPP_Sub_Domain']
df2 = df2[col2]
d = {k: g['MSPP_Sub_Domain'].unique() for k, g in df2.groupby('MSPP_Domain')}


# Store Model
def store_model(model, file_name, protocol=3):
    joblib.dump(model, '{}.pkl'.format(file_name), protocol=protocol)


import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2

Y_COLUMN = "MSPP_Sub_Domain"
TEXT_COLUMN = "IssueDesc"


def test_pipeline(df, nlp_pipeline, Domain):
    y = df[Y_COLUMN].copy()
    X = pd.Series(df[TEXT_COLUMN])
    rskf = StratifiedKFold(n_splits=5, random_state=42, shuffle = True)
    losses = []
    accuracies = []
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nlp_pipeline.fit(X_train, y_train)
        accuracies.append(metrics.accuracy_score(y_test, nlp_pipeline.predict(X_test)))
        

    #print("mean accuracy: {0}".format(round(np.mean(accuracies), 3)))
    #print("class: ", Domain)
    return round(np.mean(accuracies), 3)



unigram_log_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('logreg', linear_model.LogisticRegression())
])

ngram_pipe = Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 2))),
    ('mnb',MultinomialNB(fit_prior=True, class_prior=None))
])

LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', analyzer='word', 
                        encoding='latin-1', ngram_range=(1, 2), stop_words=remove_stop_words())),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])


rf_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', analyzer='word', 
                        encoding='latin-1', ngram_range=(1, 2), stop_words=remove_stop_words())),
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)),
            ])

svc_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', analyzer='word', 
                        encoding='latin-1', ngram_range=(1, 2), stop_words=remove_stop_words())),
    ('chi', SelectKBest(chi2, k='all')),
    ('svc', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))
    
])

classifiers = [
    ("ngram", ngram_pipe),
    ("unigram", unigram_log_pipe),
    ('log', LogReg_pipeline),
    ('rf', rf_pipeline),
    ('svm', svc_pipeline)
]

mixed_pipe = Pipeline([
    ("voting", VotingClassifier(classifiers, voting="hard"))
])

#test_pipeline(train_df, mixed_pipe)

scores = {}
for l,i in d.items():
    if (l == 'Cryptography') or (l == 'Human Resources Security'):
        scores[l] = 0
    else:
        train_df = df[df.MSPP_Domain == l].reset_index()
        acc = test_pipeline(train_df, mixed_pipe, l)
        scores[l] = acc
scores

"""
Accuracy Scores (mean):
---------------------------------------------------------
 'Access Control': 0.775,
 'Asset Management': 0.623,
 'Business Continuity Management': 0.781,
 'Communications Security': 0.717,
 'Compliance': 0.619,
 'Cryptography': 0,
 'Human Resources Security': 0,
 'Information Security Incident Management': 0.4,
 'Operations Security': 0.778,
 'Organization of Information Security': 0.778,
 'Physical and Environmental Security': 0.633,
 'Supplier Relationships': 0.317,
 'Systems Acquisition Development and Maintenance': 0.561
-----------------------------------------------------------
"""
