#!/usr/bin/python3

import pandas as pd
import numpy as np
#load the data
titanic_data = pd.read_csv("../titanic/tested.csv") 
titanic_data.pop("Cabin")

mask = titanic_data.isnull()
for coloum in titanic_data.keys():
    for i, bool in enumerate(mask[coloum]):
        if bool == True:
            titanic_data.drop(i , inplace=True)

# get the outut lable form the data
y = titanic_data.pop("Survived")
y = np.array(y).reshape(len(y), 1)

# preproecess the categorical data 
from sklearn import preprocessing

features = []
for category in titanic_data.keys():
    vocab = titanic_data[category].unique()
    vocab = vocab.reshape(len(vocab), 1)
    enc = preprocessing.OrdinalEncoder()
    enc.fit(vocab)
    features.append(enc.transform(np.array(titanic_data[category]).reshape(len(titanic_data[category]), 1)))
features_colums = np.concatenate(features, axis=1) # input feature

# splite the data into train and test sets 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_colums, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

logic_reg = LogisticRegression().fit(X_train, y_train)

pred_train = logic_reg.predict(X_train)
score_train = r2_score(y_train, pred_train)
print ("score_train = {} ".format(score_train))

pred_test = logic_reg.predict(X_test)
score_test = r2_score(y_test, pred_test)
print ("score_test = {} ".format(score_test))
