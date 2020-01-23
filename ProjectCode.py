############################################################################################
# This Python code is designed to use CNN to have the optimum architecture for predicting  #
# lung condition as being normal or having pneumonia.                                      #
# The lung images are JPG, grayscale & are distributed as follows:                         #
# Normal = 1,583 images                                                                    #
# Bacterial Pneumonia = 2,780                                                              #
# Viral Pneumonia = 1,493                                                                  #
# Total images = 5,856                                                                     #
#                                                                                          #
# Predictions:                                                                             #
# 1. Normal                                                                                #
# 2. Bacterial Pneumonia                                                                   #
# 3. Viral Pneumonia                                                                       #
#                                                                                          #
# The smallest image size of the dataset is 400x138 pixels of a pneumonia case. The images #
# are resized to 135x135 to save system resources & processing & maintaining a size that   #
# does not go below the smallest file.                                                     #
#                                                                                          #
# To test this code, download the image files from the \NORMAL & \PNEUMONIA folders from   #
# the Github repository https://github.com/alexbakus/MMAI-894-Team-Adelaide.               #
# On your computer system, create a folder hierarchy structure "C:\Queens\MMAI894" &       #
# save the two folders (\NORMAL, \PNEUMONIA) from Github in "C:\Queens\MMAI894".           #
# This code would also be available from the same Github repository.                       #
#                                                                                          #
# This Python code is part of Team Adelaide's project report for the course                #
# MMAI894 - Deep Learning under Prof Ofer Shai.                                            #
#                                                                                          #
# Seed code -                                                                              #
# Date: Jan 20, 2020; reads the JPG files, labels respectively & produces the X & y sets.  #
# Author: Francis Bello                                                                    #
############################################################################################


# Import libraries
import numpy as np
import pandas as pd
import random
# For local file system manipulation
import os
# For image operations
import cv2


##########################################################################################
# Variables declaration
##########################################################################################
DATADIR = r"C:\Queens\MMAI894"
SUBDIRS = ["NORMAL","PNEUMONIA"]
# Labels: 0 = normal, 1 = bacterial pneumonia, 2 = viral pneumonia
CATEGORIES = ["Normal","Pneumonia_Bacterial","Pneumonia_Viral"]
IMG_SIZE = 135
Training_Data = []
Error_Files = []
X = []
y = []
row_count = 10


##########################################################################################
# Methods definitions
##########################################################################################
# Method to populate  Training_Data, assigning each image according to it's classification
def create_training_data():
    for subdir in SUBDIRS:
        path = os.path.join(DATADIR, subdir)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                classifier = "Pneumonia_Bacterial" if img.upper().find(
                    "BACTERIA") > 0 else "Pneumonia_Viral" if img.upper().find("VIRUS") > 0 else "Normal"
                classifier_num = CATEGORIES.index(classifier)
                Training_Data.append([img_resized, classifier_num])
            except Exception as e:
                # Accumulate unreadable files, force loop to continue
                Error_Files.append(img)
                pass

##########################################################################################
# Returns the features & labels from a list
def create_features_labels(list_array):
    features_array = []
    label_array = []
    for features, label in list_array:
        features_array.append(features)
        label_array.append(label)
    return features_array, label_array


##########################################################################################
# Create the training data
create_training_data()

##########################################################################################
# Shuffle the order
random.shuffle(Training_Data)

print("Total record count: ", len(Training_Data))

##########################################################################################
# Split the training data into X & y
X, y = create_features_labels(Training_Data)

print("Check top %d of X & y:" % row_count)
print(X[:row_count])
print(y[:row_count])