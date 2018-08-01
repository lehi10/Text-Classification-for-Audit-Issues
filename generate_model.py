'''
This script performs the basic processing for applying a machine learning 
algorithm to a new datasheet (cvs) of IT Issues to classify them into 
the Microsoft Security Program Policy (MSPP) using Python libraries

The four steps are:
    1. Download a dataset (using pandas)
    2. Process the numeric data (using numpy and re)
    3. Train and evaluate the learners (using scikit-learn)
    4. Return results 
    (optional: currently commented out)
    5. Plot and compare results (using matplotlib)

The data imported for new classifications should use the following format
    Issue ID | Issue Description

The data training this model was IT audit issues from FY17 and FY18. The 
scripts uses prior issue descriptions and associated MSPP mapping to learn
from the frequency of certain words and letters to apply weighting and context.
A classification for each one indicating 1 of the 12 MSPP domains after the
script is ran then created a final column for the associated domain. 

This script uses three classifiers to predict the class of an IT audit issue
based on the metrics. 
'''

import pandas as pd
from io import StringIO
import re
import data_preparation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# selected model
from sklearn.svm import LinearSVC

# =====================================================================

def model_selection(features, labels):
    # Model Selection

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB

    from sklearn.model_selection import cross_val_score

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), 
        LinearSVC(), 
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]

    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns = ['model_name', 'fold_idx', 'accuracy'])
        
    model_accuracy = cv_df.groupby('model_name').accuracy.mean()
    return(model_accuracy)

def main():
    '''
    Loading Model Data and Pre Processing
    '''
    # read new dataset and perform pre-processing
    df = pd.read_csv("audita.csv")

    # data processing and clearning text from clean_text function
    df['IssueDesc'] = df['IssueDesc'].map(lambda com : data_preparation.clean_text(com))
    
    # Split the data to train and test sets:
    col = ['MSPP_Domain', 'IssueDesc']
    df = df[col]
    df = df[pd.notnull(df['IssueDesc'])]
    df.columns = ['MSPP_Domain', 'IssueDesc']

    # Set categories and factorize for sparce tables
    df['category_id'] = df['MSPP_Domain'].factorize()[0]
    category_id_df = df[['MSPP_Domain', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'MSPP_Domain']].values)


    '''
    Text Representation 
    The classifiers and learning algorithms cannot direclty process the text documents
    in original form. Therefore, for each terms in the dataset TfidfVectorizer is used to 
    perform the Term Frequency (TF), Inverse Document Frequency (IDF) (TF-IDF)
    '''
    # Initialize a TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', analyzer='word', 
                            encoding='latin-1', ngram_range=(1, 2), stop_words=remove_stop_words())

    features = tfidf.fit_transform(df.IssueDesc).toarray()
    labels = df.category_id
    

    # use the sklearn.feature_selection.chi2 to find the terms that are 
    # ... the most correlated with each of the products.
    from sklearn.feature_selection import chi2
    import numpy as np

    N = 10
    for MSPP_Domain, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(MSPP_Domain))
        print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))


    # Multi-class classifier for word cound and multinominal variants
    X_train, X_text, y_train, y_test = train_test_split(df['IssueDesc'], df['MSPP_Domain'], random_state = 0)
    count_vect = CountVectorizer()
    
    #count_vect = CountVectorizer(tokenizer = LemmaTokenizer(),)
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    from sklearn.externals import joblib

    # Save the results of the classifier and the vectorizer so that it does not need to be trained at runtime
    joblib.dump(count_vect, 'models/count_vect.pkl')
    joblib.dump(tfidf, 'models/tfidf_transformer.pkl')
    

    '''
    Part 3. Train and evaluate the learners (using scikit-learn)
    Model Evaludation by benchmarking 4 models: 
    (1) Logistic Regression, 
    (2) (Multinomial) Naive Bayes,
    (3) Linear Support Vector Machine and 
    (4) Random Forest
    '''
    # load best model Linear SVC on test and train data
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train,indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.fit(features, labels)

    # save model
    joblib.dump(model, 'models/classifier.pkl')
    
    # Evaluate Model
    from sklearn.feature_selection import chi2

    N = 5
    for MSPP_Domain, category_id in sorted(category_to_id.items()):
        indices = np.argsort(model.coef_[category_id])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
        bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
        
        print("# '{}':".format(MSPP_Domain))
        print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
        print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

    # Score MOdel

    from sklearn import metrics
    print(metrics.classification_report(y_test, y_pred, target_names=df['MSPP_Domain'].unique()))

# ============================================================================


if __name__ == '__main__':
    main()