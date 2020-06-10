from keras.applications.densenet import DenseNet121
ds= DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in ds.layers[:-4]:
    layer.trainable = False
for layer in ds.layers:
    print(layer, layer.trainable)

from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator



model = models.Sequential()


model.add(ds)

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

train_datagen = ImageDataGenerator(
                                   rescale=1./255,
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
              optimizer='adam',
              metrics=['acc'])

history = model.fit_generator(
                              train_dir,
                              steps_per_epoch=521 ,
                              epochs=10,
                              validation_data=test_dir,
                              validation_steps=148,
                              verbose=1)
import tensorflow as tf
import keras.backend as K

run_meta = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.keras.backend.set_session(sess)
    net = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
    
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
    
    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
