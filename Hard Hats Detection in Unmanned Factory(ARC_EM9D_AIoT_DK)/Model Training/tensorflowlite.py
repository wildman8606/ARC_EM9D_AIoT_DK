'''
Convert tflite to C model
xxd -i model.tflite > model.h
'''

import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout, MaxPooling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
import cv2
import os
from tensorflow.keras import optimizers
import csv
import math
import tensorflow as tf 

#Settng #Settng #Settng
epochs = 10000
nb_classes = 2
batch_size = 32
img_width = 32
img_height = 32
train_data_dir = '../Training Data/'

datagen = ImageDataGenerator()

train_generator =datagen.flow_from_directory(directory=train_data_dir,
    target_size=(img_width,img_height),
    classes=['0','1'],
    color_mode="grayscale",
    shuffle=True,
    class_mode='binary',
    batch_size=batch_size,
    subset='training')

x_train=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
y_train=np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])
print(x_train.shape)
print(y_train.shape)

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# step.1: Normalize
#int8   : -128 ~ 127
'''
def thinning(image):
    tmp = np.where(image < 210.0, 0, image)
    return np.where(image < 210.0, 0, 255)

x_train = thinning(x_train)
'''

x_train = (x_train - 128.0) / 128.0

'''
print(np.info(x_train))
x_train = x_train.astype(np.int8)
y_train = y_train.astype(np.int8)
print(np.info(x_train))
'''
print(x_train.shape)
print(y_train.shape)

# step.2: Build Model
model =Sequential()

model.add(Conv2D(8,(3,3), input_shape=(img_width, img_height, 1)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
print('model complied!!')

lr_reducer = ReduceLROnPlateau(monitor='loss',
                           factor=np.sqrt(0.1),
                           verbose=1,
                           cooldown=1,
                           patience=3,
                           min_lr=1e-10)


earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=7, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

callbacks = [lr_reducer, earlystop]

# step.3: Training Model
print('starting training....')
history = model.fit(x_train,y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  verbose=1,
                  callbacks=callbacks)

print('training finished!!')

model.evaluate(x_train, y_train)

print('saving models.h5')
model.save_weights('models.h5')

# step.4: TFLite Converter
image_shape = (img_width, img_width, 1)
def representative_dataset_gen():
    num_calibration_images = 2
    for i in range(num_calibration_images):
        image = tf.random.normal([1] + list(image_shape))
        yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

print('Done!!!')
