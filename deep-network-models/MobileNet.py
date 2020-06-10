

get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow
print(tensorflow.__version__)

IMG_WIDTH=224
IMG_HEIGHT=224
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

from keras.applications import MobileNet
from keras.models import Model
import keras

from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from keras.layers import Dense,GlobalAveragePooling2D



mobile = MobileNet(weights='imagenet',include_top=False)


for layer in mobile.layers[:-4]:
   layer.trainable = False

i=0
for layer in mobile.layers:
    i=i+1
    print(i,end=':  ')
    sp='\t\t'
    print(layer.name, sp, layer.trainable)





from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

base_model=MobileNet(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x)
preds=Dense(1,activation='sigmoid')(x)
model=Model(inputs=base_model.input,outputs=preds)




train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)



model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
model.summary()
train_dir = train_datagen.flow_from_directory(<train_folder_path>,
                                                 target_size=(224, 224),
                                                 batch_size=16,
                                                 class_mode='binary',shuffle=False)
test_dir = test_datagen.flow_from_directory(<test_folder_path>,
                                            target_size=(224, 224),
                                            batch_size=16,
                                            class_mode='binary',shuffle=False)
history = model.fit_generator(train_dir,
                              steps_per_epoch=521,
                              epochs=10,
                              validation_data=test_dir,
                              validation_steps=148)

model.save_weights("Weights_COVID19_MobileNet.h5")

import tensorflow as tf
import keras.backend as K

run_meta = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.keras.backend.set_session(sess)
    net = MobileNet(weights='imagenet',include_top=False)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
    



