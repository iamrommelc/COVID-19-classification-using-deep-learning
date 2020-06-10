

from __future__ import print_function

from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator


from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model


import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2
import time, datetime


preprocessing_function = None
base_model = None


IMAGE_SIZE    = (299, 299)
BATCH_SIZE    = 16
WEIGHTS_FINAL = 'model-inception_resnet_v2-final.h5'



from keras.applications.inception_resnet_v2 import preprocess_input
preprocessing_function = preprocess_input
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(<train_folder_path>,
                                                  target_size=IMAGE_SIZE,
                                                  class_mode='binary',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255)
valid_batches = valid_datagen.flow_from_directory(<test_folder_path>,
                                                  target_size=IMAGE_SIZE,
                                                  class_mode='binary',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)




def get_subfolders(directory):
    subfolders = os.listdir(directory)
    subfolders.sort()
    return subfolders
class_list = get_subfolders(<train_folder_path>)
num_classes=len(class_list)

x = base_model.output
x = Flatten()(x)
fc_layers=[1024,1024]
for fc in fc_layers:
  x = Dense(fc, activation='relu')(x)
  x = Dropout(0.5)(x)
i=0
for layer in base_model.layers:
    i=i+1
    print(i,end=':  ')
    sp='\t\t'
    print(layer.name, sp, layer.trainable)

predictions = Dense(1, activation='sigmoid')(x) 

finetune_model = Model(inputs=base_model.input, outputs=predictions)
for layer in finetune_model.layers[:-4]:
    layer.trainable = False

for layer in base_model.layers:
    i=i+1
    print(i,end=':  ')
    sp='\t\t'
    print(layer.name, sp, layer.trainable)

finetune_model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])

def lr_decay(epoch):
  if epoch%20 == 0 and epoch!=0:
    lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, lr/2)
    print("LR changed to {}".format(lr/2))
  return K.get_value(model.optimizer.lr)



learning_rate_schedule = LearningRateScheduler(lr_decay)

filepath=<dataset_folder_path> + "InceptionResNetV2" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

finetune_model.fit_generator(train_batches, epochs=10,
                             steps_per_epoch=521,
                             validation_data=valid_batches,
                             validation_steps=148,
                             class_weight='auto',
                             shuffle=True, callbacks=callbacks_list)


import tensorflow as tf
import keras.backend as K

run_meta = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.keras.backend.set_session(sess)
    net = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))


