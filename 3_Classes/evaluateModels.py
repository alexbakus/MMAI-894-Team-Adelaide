# importing libraries
from keras.preprocessing.image import ImageDataGenerator

# from keras.models import Sequential
'''from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as k'''
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from tensorflow.python.keras import backend as k
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import TensorBoard
from time import time
import tensorflow as tf
import datetime
import os
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

from PIL import Image
sourceSize = 'center_crop_clean_300'
img_width, img_height = 150, 150


train_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(300)+'/train'
validation_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(300)+'/val'
test_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(300)+'/test'
nb_train_samples = 10000
nb_validation_samples = 2700
epochs = 10
batch_size = 4

if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    shuffle=False,
    target_size=(img_width, img_height),
    class_mode='categorical')


top_model_dir = 'models\\'
top_model_allFiles = glob.glob('models\\modelFromJimi_20200223-011857.h5')
TopModel_preformanceList = []
print(top_model_allFiles)
for eachModel in top_model_allFiles:
    print(eachModel)
    modelName =  (eachModel.split('\\')[-1][:-20])
    #TopModel_list.append(file_.split('\\')[-1][:-20])
    print(modelName)
    modelPath = eachModel
    model = load_model(modelPath)
    print(model.summary())
    score = model.evaluate_generator(test_generator, verbose=1)
    print(score)

    #Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(test_generator.classes, y_pred)
    print(cm)
    print('Classification Report')
    target_names = ['bacteria','normal','virus']
    cr = classification_report(test_generator.classes, y_pred, target_names=target_names)
    print(cr)
    print(score)
    TopModel_preformanceList.append((modelName,cm,cr,score))

print (TopModel_preformanceList)