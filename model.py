
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from keras.layers import BatchNormalization
from keras.layers.core import Activation
import os
import librosa
import re

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

from custom_classes import DataGenerator

start_note = 0
end_note = 127
num_classes = end_note - start_note + 1
sample_rate = 22050
init_kernel_size = np.int(np.floor(sample_rate / 100))
epochs = 1000
batch_size = 128

def conv_block(feat_maps_out, prev):
    # Specifying the axis and mode allows for later merging
    prev = BatchNormalization(axis=1)(prev)
    prev = Activation('relu')(prev)
    prev = keras.layers.Conv1D(feat_maps_out, 3, padding='same')(prev)
    # Specifying the axis and mode allows for later merging
    prev = BatchNormalization(axis=1)(prev)
    prev = Activation('relu')(prev)
    prev = keras.layers.Conv1D(feat_maps_out, 3, padding='same')(prev)
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # add in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = keras.layers.Conv1D(feat_maps_out, 1, padding='same')(prev)
    return prev


def Residual(feat_maps_in, feat_maps_out, prev_layer):

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')

    return keras.layers.Add()([skip, conv]) # the residual connection


def batch_and_relu(x):
    y = keras.layers.BatchNormalization()(x)
    y = keras.layers.ReLU()(y)

    return y


def build_model(size):
    print("building model architecture...")
    size = (size, 1)
    audio = keras.layers.Input(size)

    y = keras.layers.Conv1D(48, 80, strides=4, padding='same')(audio)
    y = keras.layers.MaxPool1D(pool_size=4, strides=1, padding='same')(y)
    y = Residual(48, 48, y)
    y = Residual(48, 48, y)
    y = Residual(48, 96, y)
    y = keras.layers.MaxPool1D(pool_size=4, strides=1, padding='same')(y)
    y = Residual(48, 96, y)
    y = Residual(96, 96, y)
    y = Residual(96, 96, y)
    y = Residual(96, 96, y)
    y = keras.layers.MaxPool1D(pool_size=4, strides=1, padding='same')(y)
    y = Residual(96, 192, y)
    y = Residual(192, 192, y)
    y = Residual(192, 192, y)
    y = Residual(192, 192, y)
    y = Residual(192, 192, y)
    y = Residual(192, 192, y)
    y = keras.layers.MaxPool1D(pool_size=4, strides=1, padding='same')(y)
    y = Residual(192, 384, y)
    y = Residual(384, 384, y)
    y = Residual(384, 384, y)

    y = keras.layers.AveragePooling1D()(y)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(num_classes, activation='softmax')(y)

    model = keras.Model(inputs=audio, outputs=y)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    return model


def prepare_data():
    print("annotate data...")

    partition = dict()
    labels = {}

    data_type = 'training'
    for i in range (2):
        list_of_files = os.listdir('./data/' + data_type + '/')

        num_files_train = list_of_files.__len__()

        for i in range(num_files_train):
            if list_of_files[i] != '.DS_Store':
                current_file = list_of_files[i]

                partition.setdefault(data_type, [])
                partition[data_type].append(current_file)

                # use regex to get pitch info from filename
                pitch_regex = re.compile(r'(?<=-)(\d\d\d)(?=-)')
                current_label = int(pitch_regex.findall(current_file)[0])

                labels[current_file] = current_label
        data_type = 'validation'

    return partition, labels


def save_model(name):
    print("saving model...")
    model_json = model.to_json()
    with open("%s.json" % name, "w") as json_file:
        json_file.write(model_json)

    model.save_weights("%s.h5" % name)
    print("Saved model to disk")


def load_model(name):
    print("loading model...")
    json_file = open("%s.json" % name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tf.keras.models.model_from_json(
        loaded_model_json
    )

    loaded_model.load_weights("%s.h5" % name)
    print("Loaded model from disk")

    return loaded_model


def predict(model, x_path):
    if isinstance(x_path, str):
        x, sr = librosa.core.load(x_path)
        x = x[0]
        if x.shape[1] == 2:
            x = x[:, :-1]
        x = x[0:sample_rate, :]

    x = np.expand_dims(x, axis=0)
    y = model.predict(x)
    indices = np.argmax(y, axis=1)
    print("prediction = ")

    return indices

#_______________________________________________________________________________________________________________________

# Parameters
params = {'batch_size': batch_size,
          'dim': (sample_rate,1),
          'n_classes': num_classes,
          'shuffle': True}

# Datasets
partition, labels = prepare_data()

# Generators
training_generator = DataGenerator(partition['training'], labels, 'training', **params)
validation_generator = DataGenerator(partition['validation'], labels, 'validation', **params)

# Design model
size = len(partition)
model = build_model(sample_rate)
model.summary()

early_stopping_monitor = keras.callbacks.EarlyStopping(patience=4)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=True,
                              epochs=epochs,
                              callbacks=[early_stopping_monitor, tensorboard],
                              workers=6)


plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()