#This file uses TFIDF to capture the most unique terms for each park unit (according
#to the visitor surveys), and to make some example word clouds based on them.

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

import wordcloud
from os import path
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import matplotlib.pyplot as plt


#Import the short version of the dataset (one entry per park unit)
nps_short = pd.read_csv('data_files/data_bypark', index_col=0)


#TFIDFVectorizer: Create a vectorizer object to generate term document counts
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), token_pattern="\\b[A-Za-z][A-Za-z]+\\b")

# Get the vectors & store them in a Pandas DataFrame
vectors = tfidf.fit_transform(nps_short['SignificanceComments'])
df = pd.DataFrame(vectors.todense(), columns=[tfidf.get_feature_names()])

#Inspect the results
print "TFIDF tokens: "
print df.head()


#Find the columns with the 30 largest values, for each row (park unit), and save in a dictionary
top30_dict = {}
for i, r in df.iterrows():
    row_top30 = r.sort_values(ascending=False)[0:30]
    top30_dict[nps_short.ParkAlphaCode[i]] = row_top30


#Create column names for new df to store these results
column_names = []
column_names.append('ParkAlphaCode')
for num in range(1, 31):
    column_names.append('phrase' + str(num))

#Create list of empty lists to hold the phrases associated with each of the top 30 places
phrases = [[] for _ in range(30)]

#Iterate over the lists of top-30 values held in top30_dict.values & create new
#lists of all the top phrases, all the 2nd phrases, all the 3rd phrases, etc.
for value in top30_dict.itervalues():
    for index, phrase in enumerate(phrases):
        phrase.append(value.index[index])

#Zip these into a new dataframe with one row per park unit and one column for
#each of the top 30 phrases; inspect this dataframe
top30_df = pd.DataFrame(zip(top30_dict.keys(), *phrases), columns=column_names)
pd.options.display.max_columns = 31
print top30_df.head(20)


#To make word clouds from these top phrases, first find and remove phrases that
#are fully contained within other phrases for the same park unit

#Find and store these phrases in a new dictionary
remove_dict = {}

for i, row in top30_df.iterrows():
    to_remove = []
    row = row.iloc[1:]
    for j, _ in enumerate(row):
        for k in range(int(j) + 1, len(row)):
            if row[k] in row[j]:
                to_remove.append(row[k])
            if row[j] in row[k]:
                to_remove.append(row[j])
    remove_dict[str(i)] = set(to_remove)


#Next, generate dictionary containing lists of remaining phrases
reduced_dict = {}

for i, row in top30_df.iterrows():
    newrow = list(row.iloc[1:])
    for term in remove_dict[str(i)]:
        newrow.remove(term)
    reduced_dict[top30_df.loc[i, 'ParkAlphaCode']] = newrow

#Pickle these TFIDF terms for later use
with open('data_files/tfidf.pkl', 'wb') as file:
    pickle.dump(reduced_dict, file)


#Now make some example word clouds
#First,for Great Basin National Park
text1 = str(' '.join(reduced_dict['GRBA']))
print "Great Basin NP text: ", text1

GRBA = WordCloud(max_font_size=40, background_color='white').generate(text1)
#show
plt.imshow(GRBA)
plt.axis("off")
plt.figure()
plt.show()
#store to file
GRBA.to_file('images/GRBA.png')


#Second, for Hawaii Volcanoes National Park
text2 = str(' '.join(reduced_dict['HAVO']))
print "Hawaii Volcanoes NP text: ", text2

HAVO = WordCloud(max_font_size=40, background_color='white').generate(text2)
#show
plt.imshow(HAVO)
plt.axis("off")
plt.figure()
plt.show()
#store to file
HAVO.to_file('images/HAVO.png')


#And finally for Ellis Island
text3 = str(' '.join(reduced_dict['ELIS']))
print "Ellis Island text: ", text3

ELIS = WordCloud(max_font_size=40, background_color='white').generate(text3)
#show
plt.imshow(ELIS)
plt.axis("off")
plt.figure()
plt.show()
#store to file
ELIS.to_file('images/ELIS.png')
