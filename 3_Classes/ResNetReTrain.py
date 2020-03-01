from tensorflow.keras.applications import ResNet50V2
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
img_width, img_height = 150, 150
model_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
NAME = 'ResNet50V2Retrain_'+sourceSize+model_time


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
test_data_dir = '3_Classes/data_3C_from_center_crop_clean_300_to_300/test'
nb_train_samples = 10000
nb_validation_samples = 2700
epochs = 30
batch_size =32


if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Load the VGG model
ResNet50V2_conv = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers except the last 4 layers
for layer in ResNet50V2_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in ResNet50V2_conv.layers:
    print(layer, layer.trainable)


# Create the model
model = Sequential()

# Add the vgg convolutional base model
model.add(ResNet50V2_conv)

# Add new layers
model.add(layers.Flatten())
#model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
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
lr_string = str(0.0001).replace('.','_')
model_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
mdoel_name = 'ResNet50V2_'+str(img_width)+'_'+str(batch_size)+lr_string+'_'+ model_time
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