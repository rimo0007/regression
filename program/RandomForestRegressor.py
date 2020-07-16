#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing required packages
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import sklearn.metrics as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
#Reading data sets
train = pd.read_csv('/opt/dkube/input/mercedes-benz-train.csv')
test = pd.read_csv('/opt/dkube/input/mercedes-benz-test.csv')
print("Train shape : {}\nTest shape : {}".format(train.shape, test.shape))
train_pred = pd.get_dummies(train).drop('ID', axis=1).drop('y', axis=1)
test_pred = pd.get_dummies(test).drop('ID', axis=1)

for i in test_pred:
    if(i not in train_pred.columns.tolist()):
        train_pred[i] = 0
for i in train_pred:
    if(i not in test_pred.columns.tolist()):
        test_pred[i] = 0

#Labels
labels = np.array(train['y'])

#Features
features = train_pred
features_list = features.columns.tolist()

features = np.array(features)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.08, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

base_pred = train_labels.mean()

base_pred_error = abs(base_pred - test_labels)

print("Average baseline error : {} units".format(round(base_pred_error.mean(), 2)))

#Training model
print("Training model..")
rfr = RandomForestRegressor(n_estimators = 1000, random_state=42)
model = rfr.fit(train_features, train_labels)
print("Training complete.")

predictions = model.predict(test_features)
errors = abs(predictions - test_labels)
print("Mean absolute error : {} units".format(round(errors.mean(), 2)))

predictions = model.predict(test_pred)

#Evaluation metrics
print("Model Metrics :\n")
print("R2 Score : {}".format(sm.r2_score(train['y'].values, model.predict(train_pred))))
print("MAE : {}".format(sm.mean_absolute_error(train['y'].values, model.predict(train_pred))))
print("MSE : {}".format(sm.mean_squared_error(train['y'].values, model.predict(train_pred))))
print("MSLE : {}".format(sm.mean_squared_log_error(train['y'].values, model.predict(train_pred))))

pred = pd.DataFrame({'ID':test['ID'], 'y':predictions.tolist()})
pred

