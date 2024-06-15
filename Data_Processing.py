from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_ref = zipfile.ZipFile("/content/https://drive.google.com/file/d/1ovKWcn7LeMQq4BaJyPUEu7RogcsFXhwn/view?usp=sharing")
zip_ref.extractall("/content")
zip_ref.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random

# Paths 
DATADIR1 = r"E:\SACHIN\Project\train"
DATADIR2 = r"E:\SACHIN\Project\test"
Categories = ["FAKE", "REAL"]

training_data = []
testing_data = []
img_size = 32


# Function for data mining 
def create_data_training(dire, t_data, img_size):   # dire=>Directory, t_data=>list (training_data/testing_data)
    for category in Categories:
        path = os.path.join(dire, category)  # Joining the path DATADIR1/DATADIR2 to Categories "FAKE"/"REAL"
        class_num = Categories.index(category)   # Giving Labels (FAKE=> 0 and REAL=> 1)
        i = 0
        for img in os.listdir(path):   
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  # Use IMREAD_COLOR for RGB images
                new_array = cv2.resize(img_array, (img_size, img_size))     # Resize the array
                t_data.append([new_array, class_num])   # Append array and labels in the lists training_data and testing_data
            except Exception as e:
                i += 1   # Counting bad images for each path
                pass
        print(i, " Images not found in ",path)


# Function for Creating Pickle file
def write_file(name, list):
    pickle_out = open(name, "wb")
    pickle.dump(list, pickle_out)
    pickle_out.close()


# Function call for both directories
create_data_training(DATADIR1, training_data, 32)
create_data_training(DATADIR2, testing_data, 32)

# Shuffling the data
random.shuffle(training_data)
random.shuffle(testing_data)

X_train = []
y_train = []
X_test = []
y_test = []

# Extracting features and labels of training data
for  features, label in training_data:
    X_train.append(features)
    y_train.append(label)

# Extracting features and labels of testing data
for  features, label in testing_data:
    X_test.append(features)
    y_test.append(label)

# Converting the images to numpy array and reshape the images
X_train = np.array(X_train).reshape(-1, img_size, img_size, 3)
X_test = np.array(X_test).reshape(-1, img_size, img_size, 3)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Function call for creating pickle files
write_file("X_train.pickle", X_train)
write_file("y_train.pickle", y_train)
write_file("X_test.pickle", X_test)
write_file("y_test.pickle", y_test)