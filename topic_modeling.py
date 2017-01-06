#This file uses nltk with sklearn stopwords to tokenize the combined survey text for each NPS unit.
#Next, it uses a gensim Hierarchical Dirichlet Process (HDP) to extract topics from these texts,
#and a visualization package called pyLDAvis to display the results.
#It then uses these topics in a K-means clustering model to cluster the park units.
#Finally, principal component analysis (PCA) reduces the HDP results to 2 primary
#dimensions to facilitate visual display of the clusters.

import pandas as pd
import string
import pickle
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.util import ngrams
from gensim import corpora, models, similarities, matutils
from gensim.corpora import Dictionary, MmCorpus
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from sklearn.feature_extraction import stop_words
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %matplotlib inline

#Import the short version of the dataset (one entry per park unit)
nps_short = pd.read_csv('data_files/data_bypark', index_col=0)


#Tokenize documents
#Instantiate a tokenizer object & define stopwords
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

#Tokenize the concatenated comments for each park & add these lists to the dataframe
tokens_bypark = nps_short.SignificanceComments.apply(tokenize_doc)
nps_short['TokensByPark'] = tokens_bypark

#Pickle this tokenized df for later use
with open('data_files/nps_tokens.pkl', 'wb') as file:
    pickle.dump(nps_short, file)


#Begin HDP modeling: build dictionary & corpus
def prep_corpus(docs, no_below=5, no_above=0.5):
    """Function to prepare the corpus and dictionary"""
    print('Building dictionary...')
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    dictionary.compactify()

    print('Building corpus...')
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return dictionary, corpus

dictionary, corpus = prep_corpus(nps_short.TokensByPark)
MmCorpus.serialize('topic_files/nps_short.mm', corpus)
dictionary.save('topic_files/nps_short.dict')


#Run HDP model
hdp = models.HdpModel(corpus=corpus, id2word=dictionary)
hdp.save('topic_files/npsshort_hdp.model')


#How many topics?
print "Number of topics found: ", len(hdp.show_topics(num_topics=-1))

#Visualize the topics using the pyLDAvis package (https://github.com/bmabey/pyLDAvis)
vis_data = gensimvis.prepare(hdp, corpus, dictionary)

#To display in Jupyter notebook
#pyLDAvis.display(vis_data)

#To save to an html file
pyLDAvis.save_html(vis_data, 'topic_files/HDPmodel')


#Use topics from HDP to cluster the NPS units
# 1. Transform documents from word space to topic space
hdp_corpus = hdp[corpus]
# 2. Transform topic space into a sparse matrix encompassing all documents in the corpus
hdp_corpus_transform = matutils.corpus2csc(hdp_corpus).transpose()

#3. Perform Kmeans clustering on this matrix for different #s of K & examine inertia to choose optimal K
possible_k = range(1, 12)
def select_clusters(matrix, possible_k):
    "Function to record inertia for different K in K-means"
    inertia = []
    for k in possible_k:
        kmeans = KMeans(n_clusters=k)
        hdp_clusters = kmeans.fit_predict(matrix)
        inertia.append(kmeans.inertia_)
    return inertia

inertia_by_k = select_clusters(hdp_corpus_transform, possible_k)

plt.figure()
plt.plot(possible_k, inertia_by_k)
plt.title('Inertia for K-means clustering on HDP model')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.savefig('images/inertia.png')


#Collect clusters for n=5 (either 5 or 6 looks to be defensible here)
kmeans = KMeans(n_clusters=5)
hdp_clusters = kmeans.fit_predict(hdp_corpus_transform)

#Now create a new dataframe to hold the results, and merge with the earlier TFIDF terms
#to aid with inspecting the meaning of the clusters

#Generate TFIDF dataframe
tfidf_dict = pickle.load(open('data_files/tfidf.pkl', 'rb'))
cols = ['ParkAlphaCode', 'TFIDF_Terms']
tfidf_df = pd.DataFrame(zip(tfidf_dict.keys(), tfidf_dict.values()), columns=cols)

#Generate HDP clusters dataframe
cols = ['ParkName', 'ParkAlphaCode', 'cluster']
hdp_df = pd.DataFrame(zip(nps_short.ParkName, nps_short.ParkAlphaCode, hdp_clusters), columns=cols)

#Merge the two and inspect the result
hdp_df = hdp_df.merge(tfidf_df, on='ParkAlphaCode')
print hdp_df.head()


#Use PCA to reduce HDP results to 2 components
#1. Transform sparse matrix to dense
dense_hdp = hdp_corpus_transform.toarray()
print 'Shape of dense matrix: ', dense_hdp.shape
#2. Run the PCA
pca = PCA(n_components=2)
components = pca.fit_transform(dense_hdp)
print 'Variance explained by top two components: ', pca.explained_variance_ratio_
#3. Add results to hdp_df
component1 = []
component2 = []
for row in components:
    component1.append(row[0])
    component2.append(row[1])
hdp_df['component1'] = component1
hdp_df['component2'] = component2


#Inspect cluster 0 for obvious themes
print hdp_df[hdp_df['cluster'] == 0]

#Inspect cluster 1
print hdp_df[hdp_df['cluster'] == 1]

#Inspect cluster 2
print hdp_df[hdp_df['cluster'] == 2]

#Inspect cluster 3
print hdp_df[hdp_df['cluster'] == 3]

#Inspect cluster 4
print hdp_df[hdp_df['cluster'] == 4]

#How many park units in each cluster?
nums_by_cluster = hdp_df.groupby('cluster').count()
print nums_by_cluster

#Visualize the results. They vary somewhat each time the HDP model is run, as do
#any evident substantive patterns in the clusters. Thus, I don't label the clusters
#with substantive themes here. For one example where I did so, see the comparable
#figure in the affiliated blog post.

groups = hdp_df.groupby('cluster')
categories = ['0', '1', '2', '3', '4']
cat0 = '0'
cat1 = '1'
cat2 = '2'
cat3 = '3'
cat4 = '4'

fig = plt.figure(1, figsize=(6.5,6.5))
ax = fig.add_subplot(111)
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.component1, group.component2, marker='o', linestyle='', ms=10, label=name)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Clusters from HDP topic model')
ax.legend(categories, loc='upper right', numpoints=1, fontsize='small')
plt.show();

#save to file
fig.savefig('images/topic_clustering.png')
