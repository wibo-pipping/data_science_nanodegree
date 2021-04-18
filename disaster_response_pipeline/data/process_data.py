import pandas as pd
import click

@click.command()
@click.argument('message_file', type=click.Path(exists=True))
@click.argument('category_file', type=click.Path(exists=True))
@click.argument('database_name')
def process_data(message_file, category_file, database_name):
    print(message_file, category_file, database_name)

    return None

if __name__ == '__main__':
    process_data()


# disaster_messages.csv disaster_categories.csv DisasterResponse.db