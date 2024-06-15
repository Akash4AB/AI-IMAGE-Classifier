#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,BatchNormalization,Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def read_file(name, list):
    pickle_in = open(name, "rb")
    list = pickle.load(pickle_in)

X_train = []
y_train = []
X_test = []
y_test = []

read_file("X_train.pickle", X_train)
read_file("y_train.pickle", y_train)
read_file("X_test.pickle", X_test)
read_file("y_test.pickle",y_test)


# Spliting the validation data from training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


#model
model = Sequential()

# Convolutional Block 1
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='selu', input_shape=(64,64, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))

# Convolutional Block 2
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='selu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))

# Convolutional Block 3
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='selu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='selu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(256, activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='selu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose = 1)
model_checkpoint = ModelCheckpoint(
    filepath='/content/model.h5',  # Filepath to save the model
    monitor='val_loss',            # Metric to monitor
    save_best_only=True,           # Save only the best model
    verbose=1                      # Verbosity mode
)

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping , model_checkpoint])