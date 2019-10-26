# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:34:05 2019

@author: ckaus
"""
import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):#it will walk through all the items in the directory
        for filename in filenames:
            path = os.path.join(root, filename)#it will give the entire path name of a file

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory(r'C:\Users\ckaus\Machine_Learning_Resources\Compressed\MLCourse\emails\spam', 'spam'))
data = data.append(dataFrameFromDirectory(r'C:\Users\ckaus\Machine_Learning_Resources\Compressed\MLCourse\emails\ham', 'ham'))
print(data.head())
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'][1:750].values)#it will convert every word to respective numbers and give the word's frequency

classifier = MultinomialNB()#needs two inputs: counts and targets
targets = data['class'][1:750].values#list the frequency of spam and ham type emails
classifier.fit(counts, targets)#creates the model to check spam and ham emails
#the classifier gets trained with the datasets in dataframe on applying Naive Bayes MultinomialNB()
examples = data['message'][750:1200]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
