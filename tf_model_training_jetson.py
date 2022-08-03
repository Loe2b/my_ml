
#time tracking
import time 

fichier = open("tf_Robot_time.txt", "a")
fichier.write("\n")

tmps = time.time()
def tmp():
  global tmps
  x = time.time() - tmps
  tmps = time.time()
  return x

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

x=tmp()
fichier.write("\n import time : %f" %(x))

print(tf.__version__)
with tf.device('/gpu:3'):
    a = tf.constant(3.0)

data_dir='/grid_mnt/data__data.polcms/cms/sghosh/camdata/Augmented_dataset_bin/'
data_dir = pathlib.Path(data_dir)
print("list of folders in augmented:",os.listdir(data_dir))
image_count = len(list(data_dir.glob('*/*.png')))
print("total images:",image_count)

### load acceptable image dataset
acceptable = list(data_dir.glob('Acceptable/*'))
PIL.Image.open(str(acceptable[0]))
print("total acceptable:",len(acceptable))

### load rejectable image dataset
nonacpt = list(data_dir.glob('Nonacceptable/*'))
PIL.Image.open(str(nonacpt[0]))
print("total empty:",len(nonacpt))

batch_size = 2000
img_height = 100
img_width = 180

#import tensorflow_datasets as tfds
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  color_mode='grayscale',
  image_size=(img_height, img_width),
  batch_size=batch_size)

x=tmp()
fichier.write("\n loader setup time : %f" %(x))

class_names = train_ds.class_names
print(class_names)

### define model

num_classes = len(class_names)


#mirrored_strategy = tf.distribute.MirroredStrategy()

#with mirrored_strategy.scope():

model = Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(120, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(32, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
          #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

model.summary()

x=tmp()
fichier.write("\n model setup time : %f" %(x))

model.save('CNN_bin_reduit.h5')


## fit model

epochs=100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    shuffle=True,
    verbose=1,
    epochs=epochs
)

fichier.write("\n training time : %f"%(tmp()))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


#save values on a .txt
file = open("diagramme_val_Robot.txt", "w")
file.write("\n")

for val in loss:
  file.write("%f " %(val))
file.write("|")
for val in val_loss:
  file.write("%f " %(val))
file.write("|")
file.write("%i " %(epochs))
file.write("|")

file.close()

model.save('CNN_bin.h5')

tmp()
test_loss = model.evaluate(val_ds, verbose=2, batch_size=6)
fichier.write("testing time : %f"%(tmp()))
fichier.close()

