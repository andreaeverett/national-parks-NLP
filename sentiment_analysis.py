#This file performs a simple TextBlob sentiment analysis on the long version of
#the data (with entries for each individual survey response)

import pandas as pd
from textblob import TextBlob

nps_long = pd.read_csv('data_files/npdata_long', index_col=0)
print nps_long.info()


def sentiment_analysis(text):
    """Function to get polarity and objectivity for each entry"""
    blob = TextBlob(text)
    return blob.sentiment

#Apply to each of the comments in the dataframe and separate into polarity and subjectivity
sentiments = nps_long.SignificanceComment.apply(sentiment_analysis)
polarity = [x[0] for x in sentiments]
objectivity = [x[1] for x in sentiments]

#How does average visitor sentiment vary across different parks?
#Generate a new DF that aggregates the sentiment scores by park and sorts them in descending order
columns = ['ParkName', 'ParkAlphaCode', 'polarity', 'objectivity']
sentiment_df = pd.DataFrame(zip(nps_long.ParkName, nps_long.ParkAlphaCode, polarity, objectivity), columns=columns)
sentiment_df = sentiment_df.groupby(['ParkName']).mean().sort_values(by='polarity', ascending=False)
print sentiment_df.head(25)
