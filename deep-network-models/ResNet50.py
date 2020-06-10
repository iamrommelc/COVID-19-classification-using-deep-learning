

get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow
print(tensorflow.__version__)

IMG_WIDTH=224
IMG_HEIGHT=224
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

for layer in resnet.layers[:-4]:
   layer.trainable = False

i=0
for layer in resnet.layers:
    i=i+1
    print(i,end=':  ')
    sp='\t\t'
    print(layer.name, sp, layer.trainable)


from keras.layers import Flatten, Dense
from keras.models import Sequential

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(resnet)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()


train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)



train_dir = train_datagen.flow_from_directory(<train_folder_path>,
                                                 target_size=(224, 224),
                                                 batch_size=16,
                                                 class_mode='binary')
test_dir = test_datagen.flow_from_directory(<test_folder_path>,
                                            target_size=(224, 224),
                                            batch_size=16,
                                            class_mode='binary',shuffle=False)


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
                              train_dir,
                              steps_per_epoch=521,
                              epochs=10,
                              validation_data=test_dir,
                              validation_steps=148,
                              verbose=1)

import tensorflow as tf
import keras.backend as K

run_meta = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.keras.backend.set_session(sess)
    net = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))







