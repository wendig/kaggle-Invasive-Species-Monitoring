# -*- coding: utf-8 -*-
"""

@author: Lorand

useful links:
	keras-github:
		https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
	kaggle:
		https://www.kaggle.com/c/invasive-species-monitoring/kernels
		
thanks to finetune-vgg16-0-97-with-minimal-effort
		
"""


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from keras.callbacks import TensorBoard
from keras_tqdm import TQDMNotebookCallback
from keras import backend as K

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

from datetime import datetime
import os

import numpy as np
import pandas as pd

#run move_images.py before for the correct folder structure

vgg16 = VGG16(weights='imagenet', include_top=False)

x = vgg16.get_layer('block5_conv3').output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model_final = Model(inputs=vgg16.input, outputs=x)


for layer in vgg16.layers:
    layer.trainable = False

model_final.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# You need to have these three folders each with two subfolders for the two classes.
train_data_dir = "data/train"
validation_data_dir = "data/validate"
test_data_dir = "data/test"

# 600/450 _ 500/375 _ 400/300 _ 300/225

img_width = 600  # Change image size for training here
img_height = 450 # Change image size for training here

batch_size = 5 # i achieved good and fast results with this small minibatch size for training
batch_size_val = 400 # if Tensorflow throws a memory error while validating at end of epoch, decrease validation batch size her

# set data augmentation parameters here
datagen = ImageDataGenerator(rescale=1., 
    featurewise_center=True,
    rotation_range=10,
    width_shift_range=.1,
    height_shift_range=.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="reflect")

# normalization neccessary for correct image input to VGG16
datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)

# no data augmentation for validation and test set
validgen = ImageDataGenerator(rescale=1., featurewise_center=True)
validgen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)


train_gen = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True, 
        #save_to_dir="_augmented_images/", 
        #save_prefix="aug_"
        )

val_gen = validgen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)

test_gen = validgen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode="binary",
        shuffle=False)

train_samples = len(train_gen.filenames)
validation_samples = len(val_gen.filenames)
test_samples = len(test_gen.filenames)



now = datetime.now()

# "_tf_logs" is my Tensorboard folder. Change this to your setup if you want to use TB
logdir = "_tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
tb = TensorBoard(log_dir=logdir)

epochs=10

# I stopped training automagically with EarlyStopping after 3 consecutive epochs without improvement
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

model_final.fit_generator(train_gen, epochs=epochs, 
                          steps_per_epoch=int(train_samples/batch_size), 
                          validation_data=val_gen, 
                          validation_steps=batch_size_val, 
                          verbose=0, callbacks=[early_stopping, tb, TQDMNotebookCallback()])
						  
						  
						  
for i, layer in enumerate(model_final.layers):
   print(i, layer.name)

for layer in model_final.layers[:15]:
   layer.trainable = False
for layer in model_final.layers[15:]:
   layer.trainable = True



ow = datetime.now()

# "_tf_logs" is my Tensorboard folder. Change this to your setup if you want to use TB
logdir = "_tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
tb = TensorBoard(log_dir=logdir)

epochs=50

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

model_final.fit_generator(train_gen, epochs=epochs, 
                          steps_per_epoch=int(train_samples/batch_size), 
                          validation_data=val_gen, 
                          validation_steps=int(validation_samples/batch_size), 
                          verbose=0, callbacks=[early_stopping, tb, TQDMNotebookCallback()])

#Make predictions for test images and save as submission CSV.						  
						  


