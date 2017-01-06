import logging
import json

from flask import render_template, request
from flask_wtf import Form

import numpy as np
import pandas as pd
from wtforms import fields
from wtforms.widgets import TextArea
from wtforms.validators import Required

from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from sklearn.feature_extraction import stop_words
import string
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

from . import app, nps_short, model

logger = logging.getLogger('app')

class PredictForm(Form):
    """Fields for Predict"""
    #statement = fields.TextAreaField('Describe what you are looking for:', [validators.optional(), validators.length(max=1000)])

    #statement_text = fields.TextAreaField('Describe what you are looking for in your visit:', validators=[Required()])
    statement_text = fields.StringField('Describe what you are looking for in your visit:', widget=TextArea(), validators=[Required()])

    submit = fields.SubmitField('Submit')


tokenizer = RegexpTokenizer("[a-z][a-z]+['\w]+")
stops = set(stop_words.ENGLISH_STOP_WORDS)
additional_stopwords = ["an", "is", "its", "isnt", "dont", "doesnt", "he", "his", "my", "ve"]
stops = stops.union(additional_stopwords)

def tokenize_doc(document):
    """Function to tokenize a document"""
    word_list = []
    doc_no_apost = document.translate(None, "'")
    word_tokens = tokenizer.tokenize(doc_no_apost.lower())
    for word in word_tokens:
        if word not in stops:
            word_list.append(word)
    return word_list


def get_vectors(word_list):
    """Function to generate a unique 300-feature vector for each word in a document/word-list
    and average into a single vector to represent the document"""
    feature_vectors = []
    for word in word_list:
        try:
            word_vec = model[word]
            feature_vectors.append(word_vec)
        except:
            pass
    return np.array(pd.DataFrame(zip(*feature_vectors)).mean(axis=1))


def park_recommender(string):
    """Function that takes a text input (string) from user and identifies the five
    closest matching NPS units"""
    columns=['ParkName', 'cosine_similarity']
    word_list = tokenize_doc(string)
    string_vector = get_vectors(word_list)
    string_vector = string_vector.reshape(1, -1)

    cosine_similarity_list = []
    cosine_similarity_list = nps_short.avg_vector.apply(
                            lambda vector: cosine_similarity(np.array(vector).reshape(1, -1), string_vector))

    park_vector_pairs = pd.DataFrame({
        "ParkName": nps_short.ParkName,
        "cosine_similarity": cosine_similarity_list
    })

    sorted = park_vector_pairs.sort_values(by='cosine_similarity', ascending=False)
    first, second, third, fourth, fifth = [sorted.iloc[i, 0] for i in range(5)]
    return first, second, third, fourth, fifth


@app.route('/', methods=('GET', 'POST'))
def index():
    """Index page"""
    form = PredictForm()
    first = None
    second = None
    third = None
    fourth = None
    fifth = None


    if request.method == "POST" and form.validate_on_submit():
        # store the submitted text
        submitted_data = form.data

        # Retrieve text from form
        target_text = str(submitted_data['statement_text'])

        # Return the recommended park names
        first, second, third, fourth, fifth = park_recommender(target_text)

    return render_template('index.html',
        form=form,
        predictions=[first, second, third, fourth, fifth])
