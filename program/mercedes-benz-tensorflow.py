#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import keras
import tensorflow as tf
train_df=pd.read_csv('/opt/dkube/input/mercedes-benz-train.csv')
test_df=pd.read_csv('/opt/dkube/input/mercedes-benz-test.csv')
train_df.head()
test_df.head()
test_df.isnull().values.any()
train_df.isnull().values.any()
train_df.shape
dummy1=pd.get_dummies(train_df.iloc[:,2:10])
dummy1.shape
train_df.columns
train_df=train_df.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)
train_df.head()
# In[12]:
train_df=pd.concat([dummy1,train_df],axis=1)
train_df.head()
dummy2=pd.get_dummies(test_df.iloc[:,1:9])
dummy2.shape
test_df=test_df.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'],axis=1)
test_df.head()
test_df=pd.concat([dummy2,test_df],axis=1)
test_df.head()
from sklearn.model_selection import train_test_split
train=train_df.drop(['y'],axis=1)
test=train_df['y']
X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler().fit(X_train)
X_train=scaler1.transform(X_train)
X_test=scaler1.transform(X_test);
X_train.shape
from keras.models import Sequential
from keras.layers import Dense
NN_model = Sequential()
# The Input Layer :
NN_model.add(Dense(128,kernel_initializer='normal',input_dim =X_train.shape[1], activation='relu'))
# The Hidden Layers :
NN_model.add(Dense(256,activation='relu'))
NN_model.add(Dense(256,activation='relu'))
# The Output Layer :
NN_model.add(Dense(1,activation='linear'))
# Compile the network :
NN_model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=NN_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
print(history.history.keys())

#print(test_df.shape)
#print(train_df.shape)
#output=NN_model.predict(test_df.iloc[:,0:563])
#print(output)
import os, shutil
folder = "/opt/dkube/output/"
print(os.listdir(folder))

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
        
print(os.listdir(folder))

input_names = ['regression']
name_to_input = {name: t_input for name, t_input in zip(input_names, history.model.inputs)}

tf.saved_model.simple_save(
    keras.backend.get_session(),
    "/opt/dkube/output/"+"model-output",
    inputs=name_to_input,
    outputs={t.name: t for t in history.model.outputs})
sys.exit()

