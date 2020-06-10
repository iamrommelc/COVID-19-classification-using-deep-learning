

get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow
print(tensorflow.__version__)



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
train_dir = train_datagen.flow_from_directory(<train_folder_path>,
                                                 target_size=(299, 299),
                                                 batch_size=16,
                                                 class_mode='binary',shuffle=False)
test_dir = test_datagen.flow_from_directory(<test_folder_path>,
                                            target_size=(299, 299),
                                            batch_size=16,
                                            class_mode='binary',shuffle=False)


from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
input_shape = (299,299, 3)
nclass = len(train_dir.class_indices)

base_model = applications.InceptionV3(weights='imagenet', 
                                include_top=False, 
                                input_shape=(299,299,3))
for layer in base_model.layers[:-4]:
   layer.trainable = False

i=0
for layer in base_model.layers:
    i=i+1
    print(i,end=':  ')
    sp='\t\t'
    print(layer.name, sp, layer.trainable)

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(1, 
                    activation='sigmoid'))

model = add_model


model.compile(loss='binary_crossentropy', 
              optimizer=optimizers.SGD(lr=1e-4, 
                                       momentum=0.9),
              metrics=['accuracy'])
model.summary()

history = model.fit_generator(train_dir,
                              steps_per_epoch=521,
                              epochs=10,
                              validation_data=test_dir,
                              validation_steps=148)

model.save_weights("Weights_COVID19_InceptionV3.h5")

import tensorflow as tf
import keras.backend as K

run_meta = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.keras.backend.set_session(sess)
    net = applications.InceptionV3(weights='imagenet', 
                                include_top=False, 
                                input_shape=(299,299,3))

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.compat.v1.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
