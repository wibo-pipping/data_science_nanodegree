import sys
import click
import logging
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import pickle

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create streamhandler
ch = logging.StreamHandler()
# Set logging format and add to handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add handler to logger
logger.addHandler(ch)


URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """Read in the data from the SQLite DB from the messages table. Return the training data variable X,
    the outcome columns Y and the names of the categories

    :param database_filepath: path to the SQLite DB with the messages table
    :return: a Pandas DataFrame
    """
    db_connection_str = 'sqlite:///'+database_filepath
    logger.warning(f'connection path:{db_connection_str}')

    engine = create_engine(db_connection_str)
    logger.debug(f'Created db engine {engine}')
    df = pd.read_sql_table('messages', engine)
    logger.debug(f'Finished reading the data, {df.shape[0]} rows and {df.shape[1]} columns')

    # input message from social media
    X = df['message']

    # 36 columns
    Y = df.iloc[:,-36:] # 36 output columns, assuming the last 36 columns are the output measure
    category_names = Y.columns.tolist()
    logger.debug(f'Category names: {category_names}')

    return X, Y, category_names


def tokenize(text):
    """Input a string of text to be normalised, lemmatized and tokenized.

    :param text: string of text
    :return: list of cleaned and lemmatized word tokens
    """
    # get list of all urls using regex
    detected_urls = re.findall(URL_REGEX, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # normalize the text
    normalized_text = re.sub(r'[^a-z0-9]', " ", text).lower()

    # tokenize text
    tokens = word_tokenize(normalized_text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Define the pipeline to vectorize, transform and classify the data.

    :return: pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]
    )

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


@click.command()
@click.argument('database_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def main(database_filepath, model_filepath):
    """
    Please provide the filepath of the disaster messages database as the first argument and
    the filepath of the pickle file to save the model to as the second argument.


    \b
    Example command:
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    """
    logger.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    if p:

        logger.info('Building model...')
        model = build_model()

        logger.info('Searching through param grid and selecting best performing model')
        model = search_grid(model, X_train, Y_train)
    else:
        logger.info('Reading pickle file')
        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)

    logger.info('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    if p:
        logger.info('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        logger.info('Trained model saved!')


if __name__ == '__main__':
    main()