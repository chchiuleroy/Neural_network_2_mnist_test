# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:32:02 2017

@author: leroy
"""

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

np.random.seed(10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test4D_nor = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32"))/255
x_train4D_nor = (x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32"))/255
y_train_oh = np_utils.to_categorical(y_train)
y_test_oh = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (5, 5), padding = "same", input_shape = (28, 28, 1), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 36, kernel_size = (5, 5), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
print(model.summary())
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
train_history = model.fit(x = x_train4D_nor, y = y_train_oh, validation_split = 0.2, epochs = 12, batch_size = 300, verbose = 2)
 
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
scores = model.evaluate(x_test4D_nor, y_test_oh)
prediction = model.predict_classes(x_test4D_nor)
prediction[:10]

pd.crosstab(y_test, prediction, rownames = ["label"], colnames = ["predict"])