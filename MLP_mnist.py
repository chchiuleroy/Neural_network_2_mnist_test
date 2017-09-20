# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:35:56 2017

@author: leroy
"""

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(123456)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_n = (x_train.reshape(60000, 28*28).astype("float32"))/255
x_test_n = (x_test.reshape(10000, 28*28).astype("float32"))/255

y_train_n = np_utils.to_categorical(y_train)
y_test_n = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(units = 1000, input_dim = 784, kernel_initializer = "normal", activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units = 1000, kernel_initializer = "normal", activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units = 10, kernel_initializer = "normal", activation = "softmax"))

print(model.summary())

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

train_history = model.fit(x_train_n, y_train_n, validation_split = 0.2, epochs = 12, batch_size = 300, verbose = 2)

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show() 
    
show_train_history(train_history, "acc", "val_acc")
show_train_history(train_history, "loss", "val_loss")

scores = model.evaluate(x_test_n, y_test_n)
prediction = model.predict_classes(x_test_n)

pd.crosstab(y_test, prediction, rownames = ["label"], colnames = ["predict"])