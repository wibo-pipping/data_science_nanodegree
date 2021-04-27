import sys
import click
import logging
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


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
    pass


def build_model():
    pass


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

    logger.info('Building model...')
    model = build_model()

    logger.info('Training model...')
    model.fit(X_train, Y_train)

    logger.info('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    logger.info('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    logger.info('Trained model saved!')

if __name__ == '__main__':
    main()