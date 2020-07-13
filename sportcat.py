from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activition, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
## pip install tensorflow-datasets
import tensorflow_datasets as tfds
import tensorflow as tf
import logging
import numpy as np
import time

def dog_cat_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def dog_cat_train(model):
    splits = tfds.Split.TRAIN.subsplit(weighted=(80, 10, 10))
    (cat_train, cat_valid, cat_test), info = tfds.load('cats_vs_dogs',
        split=list(splits), with_info=True, as_supervised=True)

    def pre_process_image(image, label):
        image = tf.cast(image, tf.float32)
        image = image/255.0
        image = tf.image.resize(image, (128, 128))
        return image, label

    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000
    train_batch = cat_train.map(pre_process_image)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .repeat().batch(BATCH_SIZE)
    validation_batch = cat_valid.map(pre_process_image)
        .repeat().batch(BATCH_SIZE)

    t_start = time.time()
    model.fit(train_batch, steps_per_epoch=4000, epochs=2,
              validation_data=validation_batch,
              validation_steps=10,
              callbacks=None)
    print("Training done, dT:", time.time() - t_start)

    model = dog_cat_model()
    dog_cat_train(model)
    model.save('dogs_cats.h5')

    def dog_cat_predict(model, image_file):
        label_names = ["cat", "dog"]

        img = keras.preprocessing.image.load_img(image_file,
            target_size=(128, 128))
        img_arr = np.expand_dims(img, axis=0) / 255.0
        result = model.predict_classes(img_arr)
        print("Result: %s" % label_names[result[0][0]])

        