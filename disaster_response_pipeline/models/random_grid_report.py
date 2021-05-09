import click
import pickle
import pandas as pd
import logging

from train_classifier import tokenize

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

def read_pickle(pkl_file):
    """Read the pickle file content and return the content

    :param pkl_file: filepath to pickle file
    :return: content of pickel file
    """
    with open(pkl_file, 'rb') as f:
        content = pickle.load(f)

    return content


def create_result_df(search_dict):
    """Takes a RandomizedSearchCV dict from sklearn and splits out the results and tested parameters

    :param search_dict:RandomizedSearchCV dict object
    :return: Pandas DataFrame with rank, scores and tested parameter set
    """

    # get the results and corresponding params
    results = search_dict.cv_results_
    param_dict = results['params']

    # Create dataframe and select score columns
    result_df = pd.DataFrame(results).loc[:,['rank_test_score', 'mean_test_score', 'std_test_score']]

    # Creat param dataframe with one column per tested param
    param_df = pd.DataFrame(param_dict)
    param_df.columns = [col.split('__')[-1] for col in param_df.columns]

    # Join the results to the params and sort by best scoring to worst scoring
    df = (result_df.join(param_df).sort_values('rank_test_score'))

    return df

@click.command()
@click.argument('search_pkl', type=click.Path())
def main(search_pkl):
    """
    Please provide the filepath to the random search pickle output.

    Prints out the search grid results on commandline

    \b
    Example command:
    python random_grid_report.py random_search.pkl
    """
    logger.info(f'Reading pickle file... {search_pkl}')
    search_dict = read_pickle(search_pkl)

    logger.info('Creating result DataFrame')
    result_df = create_result_df(search_dict)

    # print params tested:
    print(result_df.to_string(index=False))

if __name__ == '__main__':
    main()