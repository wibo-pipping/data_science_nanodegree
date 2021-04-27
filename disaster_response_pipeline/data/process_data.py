import click
import logging
import pandas as pd
from sqlalchemy import create_engine


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


def load_data(messages_filepath, categories_filepath):
    """Read in messages and categories from respective files. Merges the two datasets on the id column.

    :param messages_filepath: filepath of messages dataset
    :param categories_filepath:  filepath of categories dataset
    :return: pandas DataFrame where messages and categories are merged.
    """
    logger.debug(f'Reading data from {messages_filepath}')
    messages = pd.read_csv(messages_filepath)
    logger.debug(f'Loaded {messages_filepath}, found {messages.shape[0]} lines with columns: '
                 f'{", ".join(messages.columns)}')


    logger.debug(f'Reading data from {categories_filepath}')
    categories = pd.read_csv(categories_filepath)
    logger.debug(f'Loaded {categories_filepath}, found {categories.shape[0]} lines with columns: '
                 f'{", ".join(categories.columns)}')

    logger.debug('Merging messages and categories using the "id" column...')
    df = messages.merge(categories, on='id')
    logger.debug('Done merging messages and categories')

    return df


def clean_data(df):
    """Clean the input dataframe by splitting the categories into separate columns and removing duplicates from the df

    :param df:
    :return: cleaned df
    """

    # Split out the list of categories into their own columns
    logger.info('Splitting categories on delimiter ";" to their own columns')
    category_split = df['categories'].str.split(';', expand=True)
    # Get a list of columns and set the column names on the category_split
    logger.debug('Setting split category column names')
    category_columns = category_split.iloc[0].apply(lambda x: x.split('-')[0]).tolist()
    category_split.columns = category_columns
    logger.info(f'Found {len(category_columns)} columns: {", ".join(category_columns)}')
    # Update the category values to 0 or 1 by stripping the first line
    logger.info('Cleaning the category values to be 0 or 1')
    category_split = category_split.applymap(lambda x: int(x.split('-')[1]))

    # Drop the original categories column from the df
    logger.debug('dropping the old category column')
    df = df.drop(columns='categories')

    # Concat the expanded categories back to the df
    logger.debug('Adding the split categories to df')
    df = pd.concat([df, category_split], axis=1)

    # Dropping duplicated message ids. Assuming the last record found is the latest record and is correct.
    logger.info(f'Dropping duplicate message ids, found {df.duplicated(subset="id").sum()} duplicated records...')
    df = df.drop_duplicates(subset="id", keep='last')

    logger.info(f'Dropping split categories that have values that are not equal to 1 or 0')
    # Create filter view on valid rows to include, use axis=1 to get 1 boolean value per row
    in_bounds_filter = ((df[category_columns] >= 0) & (df[category_columns] <= 1)).all(axis=1)
    logger.info(f'Keeping {in_bounds_filter.sum()} rows with inbound values,'
                f'dropping {in_bounds_filter.size-in_bounds_filter.sum()} incorrect rows')
    df = df.loc[in_bounds_filter,:]

    return df


def save_data(df, database_filename):
    """Stores the input df to a SQLite DB in a table called messages

    :param df: dataframe to be stored to database
    :param database_filename: Filepath where db should be stored.
    """

    # Assuming database_filename is correct and includes .db
    engine = create_engine(f'sqlite:///{database_filename}')
    logger.debug('Writing table messages')
    df.to_sql('messages', engine, index=False, if_exists='replace')


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

    logger.info(f'Loading data; MESSAGES: {messages_filepath}, CATEGORIES: {categories_filepath}')
    df = load_data(messages_filepath, categories_filepath)

    logger.info('Cleaning data...')
    df = clean_data(df)

    logger.info(f'Saving data to DATABASE: {database_filename}')
    save_data(df, database_filename)

    logger.info('Cleaned data saved to database!')

if __name__ == '__main__':
    main()