# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:36:49 2023

@author: Marcus
"""
#Imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#Import review file. 2 separate ones for differnt environments
df = pd.read_csv("C:/Users/marms/OneDrive - J Turner Research/Desktop/ToxicData/train.csv")
df["comment_text"].dropna(inplace= True)
df["comment_text"] = df["comment_text"].astype(str)

def target_setter(value):
    if value >= 0.6:
        return(1)
    else:
        return(0)

#Call previous function as lambda format    
df["true_target"] = df["target"].apply(lambda x: target_setter(x))

#X is feature and Y is target
X = df["comment_text"]
Y = df["true_target"]

#Split data 67-33 split chosen arbitrarily
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.33)

#Create Count Vectorizer
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(X_train)

#Create TF-IDF Transformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Create Multinomial Naive Bayes Model
model = LogisticRegression().fit(X_train_tfidf, Y_train)

#Format test data
X_test_counts = count_vector.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = model.predict(X_test_tfidf)

#Create Classification report: Overall Accuracy 0.79
target_names = ["toxic", "not-toxic"]
actual = list(Y_test)
predicted = list(predicted)
print(classification_report(actual, predicted, target_names=target_names))

test = pd.read_csv("C:/Users/marms/OneDrive - J Turner Research/Desktop/ToxicData/test.csv")