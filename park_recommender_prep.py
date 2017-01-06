#This file generates part of the data needed to run the 'Park_Recommender' app by
#using Google word vectors to generate vector representations of the pre-tokenized
#NPS survey data (yielding one vector for each NPS unit).

import pandas as pd
import numpy as np
import pickle
from gensim import models

#Open pre-tokenized data (one observation per NPS unit).
nps_short = pickle.load(open('data_files/nps_tokens.pkl', 'rb'))
print nps_short.head()

#Use gensim & Google word vectors (300 features each) to build the model I will
#apply to each word in each document (NPS unit)
model = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

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

#Apply the model and add vectors to dataframe
avg_vector_bypark = nps_short.TokensByPark.apply(get_vectors)
nps_short['avg_vector'] = avg_vector_bypark
print nps_short.head()

#Export for use in recommender app
nps_short.to_json('nps_short.json')
