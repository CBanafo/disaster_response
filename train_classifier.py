import sys
import re
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier

import sqlite3
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.engine.reflection import Inspector


database_filepath = "D:\Jupiter_files\data_science\nano_degree_files\DSND_Term2-master\project_files\Disaster_Response_Pipeline\notebooks\processed_data.db"


def load_data(database_filepath):
    """Load processed data
    
    Arguments:
        database_filepath {str} -- The filepath of SQLite database
    
    Returns:
        Dataframe -- Pandas Dataframe of processed data
    """
    # load data from database
    # engine = create_engine('sqlite:///processed_data.db')
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM df_cleaned WHERE message NOT LIKE "%chlorox%"', engine) #remove column with chorox since it was generating issues
    X = df[['message']]
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """Tokenizer for NLP pipeline
    
    Arguments:
        text {str} -- single unprocessed message
    
    Returns:
        list -- List of processed and tokenized words
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = re.sub(url, 'urlplaceholder', text)
    # text = text.replace(url, "urlplaceholder")

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize, normalize case, and remove leading/trailing white space
    clean_tok = [lemmatizer.lemmatize(tok.lower().strip()) for tok in tokens if tok not in stopwords.words("english")]

    return (clean_tok)


def build_model():
    """Build a machine learning pipeline
    
    Returns:
        object -- Grid search object
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=False)))
])
#     parameters = {'vect__binary': [True, False],
#  'clf__estimator__min_samples_leaf': [1],
#  'clf__estimator__min_samples_split': [2],
#  'clf__estimator__min_weight_fraction_leaf': [0.0],
#  'clf__estimator__n_estimators': [50,100]}

    # model = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Calculate and display the evaluation of ML model
    
    Arguments:
        model {object} -- Grid search object
        X_test {dataframe} -- Pandas Dataframe for features
        Y_test {dataframe} -- Pandas Dataframe for targets
        category_names {list} -- list of column names of targets
    """
    # Get result
    target_names = category_names
    y_pred = model.predict(X_test.message.values)
    y_pred = np.asarray(y_pred)
    y_true = np.array(Y_test)

    # Build a dataframe to store metrics and scores
    df = pd.DataFrame()
    for i,target in enumerate(target_names):
        accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        precision = precision_score(y_true[:, i], y_pred[:, i])
        recall = recall_score(y_true[:, i], y_pred[:, i])
        df = df.append({'index':target,'Accuracy':accuracy,'F1 Score':f1,'Precision':precision,'Recall':recall},ignore_index = True)
    # print results
    print(df)
    print(df.describe())


model_filepath = "D:\Jupiter_files\data_science\nano_degree_files\DSND_Term2-master\project_files\Disaster_Response_Pipeline\notebooks\disaster_response_model.pkl"


def save_model(model, model_filepath):
    # saving model
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

print(len(sys.argv))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.message.values, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

