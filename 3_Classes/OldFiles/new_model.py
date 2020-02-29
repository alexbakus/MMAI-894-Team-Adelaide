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

from PIL import Image

img_width, img_height = 400, 400

train_data_dir = 'data'+str(img_width)+'/train'
validation_data_dir = 'data'+str(img_width)+'/validation'
nb_train_samples = 7000
nb_validation_samples = 3000
epochs = 10
batch_size = 8

if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(32,
                 kernel_size = (3, 3),
                 activation='relu',
                 strides=(2, 2),
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64,
                 kernel_size = (3, 3),
                 activation='relu',
                 strides=(2, 2),
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())


model.add(Conv2D(128,
                 kernel_size = (3, 3),
                 activation='relu',
                 strides=(2, 2),
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())


model.add(Conv2D(32,
                 kernel_size = (3, 3),
                 activation='relu',
                 strides=(2, 2),
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(256,Activation('relu')))
model.add(Dropout(0.5))
model.add(Dense(3,Activation('softmax')))

model.compile(loss='categorical_crossentropy',
              optimizer= Adam(lr=0.0001),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')


model_export = 'models\\'
if not os.path.exists(model_export):
    os.makedirs(model_export)

model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "logs\\fit\\" + model_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,patience=2)
checkpoint_callback =  tf.keras.callbacks.ModelCheckpoint(filepath=(model_export + 'model_' + model_name + '.h5'),
                                                                    monitor='val_accuracy',
                                                                    save_best_only=True)
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples / batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples / batch_size,
                    callbacks=[tensorboard_callback,es_callback,checkpoint_callback])


