import json
import plotly
import pandas as pd
import logging

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
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

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
logger.info('Creating SQLite engine')
engine = create_engine('sqlite:///data/clean_messages.db')
logger.info('Reading in messages')
df = pd.read_sql_table('messages', engine)

# load model
logger.info('Unpickling the message classification model')
model = joblib.load("models/message_model.pkl")

def generate_bar_graph(x, y, title, orientation='v', label_tickangle=0):
    """Input the x and y with a x axis label and returns a Plotly graph dict for a Bar chart with counts

    :param x: List of genres to plot
    :param y: Counts of the genres
    :param title: X axis label to display. Is used in the graph title as well
    :param orientation: 'v' or 'h' for a vertical or horizontal plot
    :return: Plotly graph dict object
    """

    # To properly set the xaxis in yaxis in orientation switches, use the base
    category_axis = {'title': title, 'tickangle': label_tickangle}
    count_axis = {'title': 'Count'}

    # Handle orientation switches
    if orientation == 'v':
        x_values = x
        y_values = y
        xaxis = category_axis
        yaxis = count_axis
    else: # Flip the x and y
        x_values = y
        y_values = x
        xaxis = count_axis
        yaxis = category_axis


    layout = {
        'title': f'Distribution of Message {title}',
        'xaxis':xaxis,
        'yaxis':yaxis
    }

    graph = {
        'data': [
            Bar(x=x_values,
                y=y_values,
                orientation=orientation
            )
        ],
        'layout':layout
    }

    return graph


def generate_pie_chart(labels, values):
    """Generate the Pie chart graph object

    :param labels: labels to use in Pie chart
    :param values: Values to use in Pie chart
    :return: Plotly graph object for Pie chart
    """
    graph = {
        'data': [
            Pie(labels=labels,
                values=values
                )
        ],
        'layout': {
            'title': f'No, One or Multi label messegas'
        }
    }

    return graph

def get_label_class(input):
    """Input an integer and returns the label class, 'No label', 'One label' or 'Multi label'

    :param input: Integer
    :return: Label class as a string
    """
    if input == 0:
        return 'No label'
    elif input == 1:
        return 'One label'
    else:
        return 'Multi label'


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Add count per label value
    label_subset = df.iloc[:, 4:]
    label_counts = label_subset.sum().sort_values()
    label_names = list(label_counts.index.str.replace('_',' '))

    # Check no labels, 1 label or multi label by summing over rows, getting the labels and counting occurances:
    multi_label_values = label_subset.sum(axis=1).apply(get_label_class).value_counts()
    multi_label_labels = list(multi_label_values.index)

    # create visuals
    graphs = [
        generate_bar_graph(genre_names, genre_counts, "Genre"),
        generate_pie_chart(multi_label_labels, multi_label_values),
        generate_bar_graph(label_names, label_counts,"Labels", label_tickangle=-45),
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('main.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()