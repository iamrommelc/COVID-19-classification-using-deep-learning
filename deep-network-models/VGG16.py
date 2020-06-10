
import tensorflow
print(tensorflow.__version__)

from keras.applications import VGG16
vgg_16= VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg_16.layers[:-4]:
   layer.trainable = False


for layer in vgg_16.layers:
    print(layer, layer.trainable)

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


model = models.Sequential()


model.add(vgg_16)

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))



model.summary()

train_datagen = ImageDataGenerator(
                                   rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


train_dir = train_datagen.flow_from_directory(<train_folder_destination>,
                                                 target_size=(224, 224),
                                                 batch_size=16,
                                                 class_mode='binary')
test_dir = test_datagen.flow_from_directory(<test_folder_destination>,
                                            target_size=(224, 224),
                                            batch_size=16,
                                            class_mode='binary',shuffle=False)


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(train_dir,
                              steps_per_epoch=245,
                              epochs=10,
                              validation_data=test_dir,
                              validation_steps=70,
                              verbose=1)

import tensorflow as tf
import keras.backend as K

run_meta = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.keras.backend.set_session(sess)
    net = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

