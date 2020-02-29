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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import TensorBoard
from time import time
import tensorflow as tf
import datetime
import os
import glob
import re
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
img_width, img_height = 300, 300


'''if not os.path.exists(log_dir):
    os.makedirs(log_dir)
'''
model_export = 'models\\'
if not os.path.exists(model_export):
    os.makedirs(model_export)

allFiles = glob.glob('models\\*h5')
trainedModel_list = []
for file_ in allFiles:
    #print (file_.split('\\')[-1][:-20])
    trainedModel_list.append(file_.split('\\')[-1][:-20])

top_model_dir = 'models\\TopModels\\'
top_model_allFiles = glob.glob('models\\TopModels\\*h5')
TopModel_list = []
for file_ in top_model_allFiles:
    print (file_.split('\\')[-1][:-20])
    TopModel_list.append(file_.split('\\')[-1][:-20])

train_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(300)+'/train'
validation_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(300)+'/val'
test_data_dir = '3_Classes/data_3C_from_'+sourceSize+'_to_'+str(300)+'/test'

trainning_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelPreformanceLog = open('model_preformance_log_'+trainning_time+'.log','w')
modelPreformanceLog.close()

nb_train_samples = 15000
nb_validation_samples = 5000
epochs = 30
batch_size = 32

if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model_performance_list = []

for eachTopModel in top_model_allFiles:
    modelNamePart = eachTopModel.split('\\')[-1][9:-20]
    print(modelNamePart)
    trainedModel_list.append(modelNamePart)
    modelArch_info = re.findall(
        r'(.*?)_conv_(.*?)_nodes_(.*?)_dense_(.*?)_kernelSz_(.*?)_optimizer_(.*?)lr_(.*?)_dropr_', modelNamePart)
    print(modelArch_info[0])

    conv_layer = int(modelArch_info[0][0])
    layer_size = int(modelArch_info[0][1])
    dense_layer = int(modelArch_info[0][2])
    kernel_size = int(modelArch_info[0][3])
    optimizer = modelArch_info[0][4]
    learn_rate = float((modelArch_info[0][5]).replace('_', '.'))
    dropout_rate = float((modelArch_info[0][6]).replace('_', '.'))

    print(conv_layer, layer_size, dense_layer, kernel_size, optimizer, learn_rate, dropout_rate)
    model_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    NAME = modelNamePart+model_time+'_T'
    NAME = NAME.replace('-', '_').replace('.', '_')
    print(NAME)


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
        model.add(Conv2D(layer_size * (2 ** l),
                         kernel_size=(kernel_size, kernel_size),
                         activation='relu',
                         padding='valid',
                         kernel_initializer='random_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())

    for _ in range(dense_layer):
        model.add(Dense(layer_size * 16))
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  lr=learn_rate,
                  metrics=['accuracy'])

    log_dir = "logs\\fit\\" + NAME
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=2)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(model_export + 'ReTrained_' + NAME + '.h5'),
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

    val_datagen = ImageDataGenerator(rescale=1. / 255,
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

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        shuffle=False,
        target_size=(img_width, img_height),
        class_mode='categorical')

    # Confution Matrix and Classification Report
    print(model.summary())
    score = model.evaluate_generator(test_generator, verbose=1)
    print(score)

    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(test_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(test_generator.classes, y_pred)
    print(cm)
    print('Classification Report')
    target_names = ['bacteria', 'normal', 'virus']
    cr = classification_report(test_generator.classes, y_pred, target_names=target_names)
    print(cr)
    print(score)

    modelPreformanceLog = open('model_preformance_log_' + trainning_time + '.log', 'a')
    modelPreformanceLog.write(NAME+','+str(score)+','+str(cm)+','+str(cr)+'\n')
    modelPreformanceLog.close()

