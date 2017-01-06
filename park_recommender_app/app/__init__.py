import logging
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from gensim import models
from flask import Flask

# create logger for app
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)

app = Flask(__name__)
app.config.from_object("app.config")

# import my model
model = models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
nps_short = pd.read_json("models/nps_short.json")

from .views import *   # flake8: noqa


# Handle Bad Requests
@app.errorhandler(404)
def page_not_found(e):
    """Page Not Found"""
    return render_template('404.html'), 404
