# importing libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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



from PIL import Image
sourceSize = 'center_crop_clean_300'
img_width, img_height = 200, 200

model_export = 'models\\'
if not os.path.exists(model_export):
    os.makedirs(model_export)

allFiles = glob.glob('models\\*h5')
trainedModel_list = []
for file_ in allFiles:
    print (file_.split('\\')[-1][:-20])
    trainedModel_list.append(file_.split('\\')[-1][:-20])

train_data_dir = '3_Classes/data_3C_from_center_crop_clean_300_to_300/train'
validation_data_dir = '3_Classes/data_3C_from_center_crop_clean_300_to_300/val'
nb_train_samples = 10000
nb_validation_samples = 2700
epochs = 15
batch_size = 32

if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



dense_layers = [0, 1, 2]
layer_sizes = [16, 32]
conv_layers = [3,4,5]
learn_rates = [0.001, 0.01]
optimizers = ['adam','sgd']
kernel_sizes = [2, 3, 4]
dropout_rates = [0.0, 0.25,0.5]
#momentums = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for kernel_size in kernel_sizes:
                for optimizer in optimizers:
                    for learn_rate in learn_rates:
                        for dropout_rate in dropout_rates:
                            model_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                            NAME = "{}-conv-{}-nodes-{}-dense-{}-kernelSz-{}-optimizer-{}lr-{}-dropr-{}-T".format(conv_layer,
                                                                                                                      layer_size,
                                                                                                                      dense_layer,
                                                                                                                      kernel_size,
                                                                                                                      optimizer,
                                                                                                                      learn_rate,
                                                                                                                      dropout_rate,
                                                                                                                            model_time)
                            NAME = NAME.replace('-', '_').replace('.', '_')
                            print(NAME)

                            checkNAME = "{}-conv-{}-nodes-{}-dense-{}-kernelSz-{}-optimizer-{}lr-{}-dropr-".format(conv_layer,
                                                                                                                    layer_size,
                                                                                                                    dense_layer,
                                                                                                                    kernel_size,
                                                                                                                    optimizer,
                                                                                                                    learn_rate,
                                                                                                                    dropout_rate)
                            checkNAME = NAME.replace('-', '_').replace('.', '_')
                            print(checkNAME)
                            # print(NAME)
                            checkName = 'NewModel_' + checkNAME + '.h5'
                            if checkName[:-20] in trainedModel_list:
                                print(checkName, ' model trainded before, skipped')
                            else:
                                print(NAME, 'in progress...')

                                model = Sequential()

                                model.add(Conv2D(layer_size,
                                                 kernel_size=(kernel_size, kernel_size),
                                                 activation='relu',
                                                 padding='valid',
                                                 kernel_initializer='random_uniform',
                                                 input_shape=input_shape))
                                model.add(MaxPooling2D(pool_size=(2, 2)))
                                model.add(Dropout(dropout_rate))

                                for l in range(conv_layer - 1):
                                    model.add(Conv2D(layer_size*(2**l),
                                                     kernel_size=(kernel_size, kernel_size),
                                                     activation='relu',
                                                     padding='valid',
                                                     kernel_initializer='random_uniform'))
                                    model.add(MaxPooling2D(pool_size=(2, 2)))
                                    model.add(Dropout(dropout_rate))

                                model.add(Flatten())

                                for _ in range(dense_layer):
                                    model.add(Dense(layer_size*16))
                                    model.add(Activation('relu'))
                                    model.add(Dropout(dropout_rate))

                                model.add(Dense(3, activation='softmax'))

                                model.compile(loss='categorical_crossentropy',
                                              optimizer = optimizer,
                                              lr = learn_rate,
                                              metrics=['accuracy'])




                                log_dir = "logs\\fit\\" + NAME

                                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                                es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,patience=2)
                                checkpoint_callback =  tf.keras.callbacks.ModelCheckpoint(filepath=(model_export + 'NewModel_' + NAME + '.h5'),
                                                                                                    monitor='val_accuracy',
                                                                                                    save_best_only=True)

                                print(model.summary())

                                train_datagen = ImageDataGenerator(
                                    rescale=1. / 255,
                                    shear_range=0.2,
                                    rotation_range=40,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True)

                                val_datagen = ImageDataGenerator(
                                    rescale=1. / 255,
                                    shear_range=0.2,
                                    rotation_range=40,
                                    zoom_range=0.2,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    horizontal_flip=True)

                                train_generator = train_datagen.flow_from_directory(
                                    train_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size, class_mode='categorical')

                                validation_generator = val_datagen.flow_from_directory(
                                    validation_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size, class_mode='categorical')

                                model.fit_generator(train_generator,
                                                    steps_per_epoch=nb_train_samples // batch_size,
                                                    epochs=epochs,
                                                    validation_data=validation_generator,
                                                    validation_steps=nb_validation_samples // batch_size,
                                                    callbacks=[tensorboard_callback, es_callback, checkpoint_callback])


