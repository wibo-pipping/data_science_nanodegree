import pandas as pd
import click


def load_data(messages_filepath, categories_filepath):
    pass


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass


@click.command()
@click.argument('messages_filepath', type=click.Path(exists=True))
@click.argument('categories_filepath', type=click.Path(exists=True))
@click.argument('database_filename')
def main(messages_filepath, categories_filepath, database_filename):
    """
    Please provide the filepaths of the messages and categories datasets as the first and second argument respectively,
    as well as the filepath of the database to save the cleaned data to as the third argument.


    \b
    Example command:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    """
    print(messages_filepath, categories_filepath, database_filename)

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filename)

    print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()