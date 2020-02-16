from tensorflow.keras.applications import InceptionV3
# importing libraries
from keras.preprocessing.image import ImageDataGenerator

# from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from tensorflow.python.keras import backend as k
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
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
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)





sourceSize = 'center_crop_clean_300'
img_width, img_height = 300, 300
model_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
NAME = 'InceptionV3Retrain_'+sourceSize+model_time


model_export = 'models\\'
if not os.path.exists(model_export):
    os.makedirs(model_export)

allFiles = glob.glob('models\\*h5')
trainedModel_list = []
for file_ in allFiles:
    print (file_.split('\\')[-1][:-20])
    trainedModel_list.append(file_.split('\\')[-1][:-20])

train_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(img_width)+'/train'
validation_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(img_width)+'/validation'
nb_train_samples = 10000
nb_validation_samples = 2700
epochs = 15
batch_size = 16


if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Load the VGG model
ipv3_conv = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)


# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in ipv3_conv.layers[:-4]:
   layer.trainable = False


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
ipv3_conv.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# Check the trainable status of the individual layers
for layer in ipv3_conv.layers:
    print(layer, layer.trainable)


# Create the model
model = Sequential()

# Add the vgg convolutional base model
model.add(ipv3_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

# Show a summary of the model. Check the number of trainable parameters

log_dir = "logs\\fit\\" + NAME

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,patience=2)
checkpoint_callback =  tf.keras.callbacks.ModelCheckpoint(filepath=(model_export + 'model_' + NAME + '.h5'),
                                                                    monitor='val_loss',
                                                                    save_best_only=True)

print(model.summary())

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rescale=1. / 255,
                                   shear_range=0.2,
                                   rotation_range=40,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=[tensorboard_callback, es_callback, checkpoint_callback])