# classify.py
"""
*** IMPORTANT WHEN IMPORTING NEW ISSUES SAVE FILE IN IMPORT FOLDER AS:
          --------------------------------------------------      
               ***      "new_import.csv"   ****
          --------------------------------------------------
This module classifies new issues using the pickle files from the generate_model.py
Models are located in the model folder.
Now classified issues are found in the results folder.
"""
import data_preparation
from domains import domains
from sklearn.externals import joblib
import pandas as pd

def run_classification(new_df):
    model = joblib.load('models/classifier.pkl')
    count_vect = joblib.load('models/count_vect.pkl')
    tfidf = joblib.load('models/tfidf_transformer.pkl')

    # Import new files and clean them
    #new_df = pd.read_csv('new_import.csv')
    new_df['IssueDesc'] = new_df['IssueDesc'].map(lambda com : data_preparation.clean_text(com))

    # Because Naive Bayes maps its results to integers, it's necessary to map the domains codes to ints
    # len(domains) + 1 must equal "undetermined" (WIP)

    #domains = {13: "undetermined"}
    domain_mapping = {}

    for index, domain in enumerate(domains):
        domain_mapping[index] = domain

    text_features = tfidf.transform(new_df.IssueDesc)
    predictions = model.predict(text_features)
    for text, predicted in zip(new_df.IssueDesc, predictions):
        new_df['MSPP_Domain'] = [domain_mapping[p] for p in predictions]

    # save file in results folder
    return new_df
