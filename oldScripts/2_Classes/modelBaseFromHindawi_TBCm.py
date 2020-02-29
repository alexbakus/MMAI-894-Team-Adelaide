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

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


from PIL import Image
sourceSize = 'center_crop_400'
img_width, img_height = 200, 200


train_data_dir = '2_Classes/data_2C_from_'+sourceSize+'_to_'+str(img_width)+'/train'
validation_data_dir = '2_Classes/data_2C_from_'+sourceSize+'_to_'+str(img_width)+'/validation'
nb_train_samples = 14000
nb_validation_samples = 6000
epochs = 10
batch_size = 32

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure


if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)





model = Sequential()

model.add(Conv2D(32,
                 kernel_size = (3, 3),
                 activation='relu',
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())

model.add(Conv2D(64,
                 kernel_size = (3, 3),
                 activation='relu',
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())


model.add(Conv2D(128,
                 kernel_size = (3, 3),
                 activation='relu',
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())


model.add(Conv2D(128,
                 kernel_size = (3, 3),
                 activation='relu',
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512,Activation('relu')))

model.add(Dense(512,Activation('relu')))

model.add(Dense(1,Activation('sigmoid')))

model.compile(loss='binary_crossentropy',
              optimizer= Adam(lr=0.0001),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=40,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')


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
print(model.summary())

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=[tensorboard_callback,es_callback,checkpoint_callback])


#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['normal', 'pneumonia']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))