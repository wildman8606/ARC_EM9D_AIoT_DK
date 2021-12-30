'''
這是沒有INT8正規化後的，但準確率較高
'''
import os
# 防止Tensorflow運行GPU內存不足造成Error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #選擇gpu
config = ConfigProto()
config.allow_soft_placement=True #如果你指定的設備不存在，允許TF自動分配
config.gpu_options.per_process_gpu_memory_fraction=0.9  #分配百分之90
config.gpu_options.allow_growth = True   #按需分配顯存，重要
session = InteractiveSession(config=config)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout,MaxPooling2D
from tensorflow import keras
from tensorflow.keras.models import load_model
import random
import cv2
from tensorflow.keras import optimizers
import csv

#Settng #Settng #Settng
batch_size = 512 #64 #1024
epochs = 10000
nb_classes = 2
batch_size = 32
img_size = 150


from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import math
import tensorflow as tf 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Input

import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_width = 32
img_height = 32
train_data_dir = './'
valid_data_dir = './'

datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.1)

train_generator =datagen.flow_from_directory(directory=train_data_dir,
    target_size=(img_width,img_height),
    classes=['0','1'],
    color_mode="grayscale",
    shuffle=True,
    class_mode='binary',
    batch_size=batch_size,
    subset='training')

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
    target_size=(img_width,img_height),
    classes=['0','1'],
    color_mode="grayscale",
    shuffle=True,
    class_mode='binary',
    batch_size=batch_size,
    subset='validation')


test_generator = validation_generator


x_train=np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
print(x_train.shape)


# step-2 : build model

model =Sequential()

model.add(Conv2D(8,(3,3), input_shape=(img_width, img_height, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(16,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy']) #rmsprop #Adam(1e-2)
print('model complied!!')

lr_reducer = ReduceLROnPlateau(monitor='loss', #val_loss
                           factor=np.sqrt(0.1),
                           verbose=1,
                           cooldown=1,
                           patience=3,
                           min_lr=1e-10)



earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

callbacks = [lr_reducer, earlystop]




print('starting training....')
training = model.fit_generator(generator=train_generator,epochs=100,validation_data=validation_generator, callbacks=callbacks)

print('training finished!!')
result = model.evaluate(test_generator)

