#model inspired from
#https://www.hindawi.com/journals/jhe/2019/4180949/
# importing libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5800)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



trainning_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelPreformanceLog = open('Top_model_preformance_log_'+trainning_time+'.log','w')
modelPreformanceLog.close()

conv_layer = 5
layer_size = 32
kernel_size = 4
kernel_initializer = 'random_uniform'
dense_layer = 1
from PIL import Image
sourceSize = 'center_crop_clean_300'
imgSize_list = [150,200,250,300]
batch_size_list =[64,32,16,12,8]
#learning_rate_list = [0.0001,0.001,0.01,0.1,1]
learning_rate_list = [0.001]
for eachLR in learning_rate_list:
    batch_size = 32
    img_width, img_height = 300, 300

    train_data_dir = '3_Classes/data_3C_from_center_crop_clean_300_to_300/train'
    validation_data_dir = '3_Classes/data_3C_from_center_crop_clean_300_to_300/val'
    test_data_dir = '3_Classes/data_3C_from_center_crop_clean_300_to_300/test'

    nb_train_samples = 10000
    nb_validation_samples = 2700
    epochs = 30
    #batch_size =32



    if k.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()

    model = Sequential()

    model.add(Conv2D(layer_size,
                     kernel_size=(kernel_size, kernel_size),
                     activation='relu',
                     padding='valid',
                     kernel_initializer='random_uniform',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for l in range(conv_layer - 1):
        model.add(Conv2D(layer_size * (2 ** l),
                         kernel_size=(kernel_size, kernel_size),
                         activation='relu',
                         padding='valid',
                         kernel_initializer='random_uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    for _ in range(dense_layer):
        model.add(Dense(layer_size * 16))
        model.add(Activation('relu'))

    model.add(Dense(3, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer= 'adam',
                  lr=eachLR,
                  metrics=['accuracy'])

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


    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        shuffle=False,
        target_size=(img_width, img_height),
        class_mode='categorical')

    model_export = 'models\\'
    if not os.path.exists(model_export):
        os.makedirs(model_export)
    lr_string = str(eachLR).replace('.','_')
    model_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mdoel_name = 'BestModel'+str(img_width)+'_'+str(batch_size)+lr_string+'_'+ model_time
    log_dir = "logs\\fit\\" + mdoel_name

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,patience=2)
    checkpoint_callback =  tf.keras.callbacks.ModelCheckpoint(filepath=(model_export + mdoel_name + '.h5'),
                                                                        monitor='val_accuracy',
                                                                        save_best_only=True)
    print(model.summary())
    print (len(train_generator),nb_train_samples // batch_size)

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        callbacks=[tensorboard_callback,es_callback,checkpoint_callback])


    #Confution Matrix and Classification Report
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

    modelPreformanceLog = open('Best_model_preformance_log_' + trainning_time + '.log', 'a')
    modelPreformanceLog.write(mdoel_name+','+str(score)+','+str(cm)+','+str(cr)+'\n')
    modelPreformanceLog.close()
