
#This file takes the survey response data from the original Excel file and transforms
#it into two CSV files suitable for NLP analysis.

import pandas as pd
import numpy as np
import pickle

#Read in the Excel file
data_dict = pd.read_excel('data_files/NPS_SignificanceUnderstanding.xlsx', sheetname=None)

#Delete the Excel sheet that summarizes the data
del data_dict['SUMMARY']

#Concatenate the lines of the remaining Excel sheets (one sheet per NPS unit)
data = pd.concat(data_dict.values())


#Drop irrelevant & incompletely filled in columns ('Updated Significance Code', 'Updated Significance Keywords')
data = data.iloc[:, 0:4]

#Drop rows that are missing the Visitor's significance comment
data = data[pd.notnull(data["Visitor's Significance Comment"])]
data = data.reset_index()
data = data.drop('index', axis=1)

#Rename columns
data.columns = ['ParkName', 'ParkAlphaCode', 'SignificanceComment', 'SignificanceCode']


print "Summary of Data by Individual Survey Response: "
print data.info()
print data.head()

#How many survey responses are there for each park unit?
counts = data.groupby(by='ParkAlphaCode').sum()
print counts.head()
max = np.max(counts)
print 'Maximum number of survey responses: ', max
min = np.min(counts)
print 'Minimum number of survey responses: ', min

#Which units saw the least / most survey responses?
sorted_counts = counts.sort_values(by='SignificanceCode')
print sorted_counts


#Save this as a CSV for use with any analyses where I want to examine the individual survey response
data.to_csv('data_files/npdata_long')


#Now create a second dataframe that sums all the comments for each unique park unit. Use this when main interest is the park units themselves.
def combine_texts(grp):
    """Function to concatenate all entries under 'SignificanceComment' together for each park unit"""
    return grp.SignificanceComment.astype(str).str.cat(sep = " ")

#Apply combine_texts function to the groups defined by each ParkAlphaCode
combined_comments = data.groupby(['ParkAlphaCode']).apply(combine_texts)

#Create dataframe with only one entry per park unit & merge it with the combined_comments series
data_bypark = data[['ParkName', 'ParkAlphaCode']].drop_duplicates()
data_bypark = data_bypark.merge(pd.DataFrame(combined_comments), left_on='ParkAlphaCode', right_index=True)
data_bypark.columns = ['ParkName', 'ParkAlphaCode', 'SignificanceComments']
data_bypark = data_bypark.reset_index()
data_bypark = data_bypark.drop('index', axis=1)

print "Summary of Data by Park Unit"
print data_bypark.info()

#Export this to its own CSV
data_bypark.to_csv('data_files/data_bypark')
