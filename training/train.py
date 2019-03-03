import cv2
from time import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import pandas as pd

# dimensions of our images.
img_width, img_height = 455, 700

csv = pd.read_csv(r"./images.csv")

# train_data_dir = 'new_data/training'
# validation_data_dir = 'new_data/validation'
nb_train_samples = int(csv.category.count() * 0.85)
nb_validation_samples = int(csv.category.count() * 0.15)
epochs = 70
batch_size = 8

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    fill_mode='nearest',
    rescale=1. / 255)

test_datagen = ImageDataGenerator(
    fill_mode='nearest',
    rescale=1. / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=csv,
    directory="./dataset",
    x_col="image",
    y_col="category",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

label_map = (train_generator.class_indices)
print(label_map)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=csv,
    directory="./dataset",
    x_col="image",
    y_col="category",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# We stellen een checkpunt in
filepath="3mar-weights.{epoch:02d}-{val_acc:.2f}.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

callbacks_list = [checkpoint, tensorboard]

# We trainen het model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    workers=4,
    callbacks=callbacks_list)

model.save('3mar.h5')