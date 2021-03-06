
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import os
import re
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import librosa

from custom_classes import DataGenerator

start_note = 0
end_note = 127
num_classes = end_note - start_note + 1
sample_rate = 22050
init_kernel_size = np.int(np.floor(sample_rate / 100))
epochs = 9
batch_size = 128


def residual_block(y, nb_channels_in, nb_channels_out):

    shortcut = y

    y = keras.layers.Conv1D(nb_channels_in,
                            kernel_size=3,
                            strides=1,
                            padding='same')(y)
    y = batch_and_relu(y)

    y = keras.layers.Conv1D(nb_channels_out,
                            kernel_size=3,
                            strides=1,
                            padding='same')(y)

    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = keras.layers.BatchNormalization()(y)

    y = keras.layers.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = keras.layers.ReLU()(y)

    return y


def batch_and_relu(x):
    y = keras.layers.BatchNormalization()(x)
    y = keras.layers.ReLU()(y)

    return y


def build_model(size):
    print("building model architecture...")
    size = (size, 1)
    audio = keras.layers.Input(size)

    # conv1
    x = keras.layers.Conv1D(filters=48,
                            kernel_size=init_kernel_size,
                            strides=4)(audio)
    x = batch_and_relu(x)
    x = keras.layers.MaxPool1D(pool_size=4)(x)

    # conv2
    for i in range(3):
        x = keras.layers.Conv1D(filters=48,
                                kernel_size=3,
                                strides=1,
                                padding="same")(x)
        x = batch_and_relu(x)
    x = keras.layers.MaxPool1D(pool_size=4)(x)

    # conv3
    for i in range(4):
        x = keras.layers.Conv1D(filters=96,
                                kernel_size=3,
                                strides=1,
                                padding="same")(x)
        x = batch_and_relu(x)
    x = keras.layers.MaxPool1D(pool_size=4)(x)

    # conv4
    for i in range(6):
        x = keras.layers.Conv1D(filters=192,
                                kernel_size=3,
                                strides=1,
                                padding="same")(x)
        x = batch_and_relu(x)
    x = keras.layers.MaxPool1D(pool_size=4)(x)

    # conv5
    for i in range(3):
        x = keras.layers.Conv1D(filters=384,
                                kernel_size=3,
                                strides=1,
                                padding="same")(x)
        x = batch_and_relu(x)

    x = keras.layers.AveragePooling1D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=audio, outputs=x)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def annotate_test_data():
    print("annotate test_data...")

    test_labels = []

    list_of_files = os.listdir('./data/' + 'test/')
    num_files = list_of_files.__len__()
    test_data = np.zeros([sample_rate, num_files])

    for i in range(num_files):
        if list_of_files[i] != '.DS_Store':
            current_file = list_of_files[i]

            # use regex to get pitch info from filename
            pitch_regex = re.compile(r'(?<=-)(\d\d\d)(?=-)')
            current_label = int(pitch_regex.findall(current_file)[0])

            test_labels.append(current_label)
            audio_data = librosa.core.load('./data/' + 'test/' + current_file)[0]
            audio_data = audio_data[0:sample_rate]
            test_data[:, i] = librosa.util.normalize(audio_data)
            print("annotate" + str(i) + "/" + str(num_files))

    np.save("test_data", test_data)
    np.save("test_labels", test_labels)

def predict(model, data, is_path):
    x = np.zeros([sample_rate, 1])
    if is_path:
        audio = librosa.core.load(data)[0]
        x[:, 0] = audio[0:sample_rate]
    else:
        x[:, 0] = data
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    y = model.predict(x)
    indices = np.argmax(y, axis=1)
    return indices


def evaluate(model):
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_labels.npy')
    scores = []

    for instance in range(test_labels.shape[0]):
        prediction = predict(model, test_data[:, instance], 0)
        print("prediction=" + str(prediction))
        print("target=" + str(test_labels[instance]))
        deviation = abs(test_labels[instance] - prediction)
        accuracy = 100 - ((deviation/127)*100)
        print("accuracy=" + str(accuracy) + "%")
        scores.append(accuracy)

    print("median= " + str(np.mean(scores)))


def prepare_data():
    print("annotate data...")

    partition = dict()
    labels = {}

    data_type = 'training'
    for i in range(2):
        list_of_files = os.listdir('E:/datasets/synthetic_note_dataset/' + data_type + '/')

        num_files_train = list_of_files.__len__()

        for i in range(num_files_train):
            if list_of_files[i] != '.DS_Store':
                current_file = list_of_files[i]
    
                partition.setdefault(data_type, [])
                partition[data_type].append(current_file)

                # use regex to get pitch info from filename
                pitch_regex = re.compile(r'(?<=sin)(\d+)')
                current_label = int(pitch_regex.findall(current_file)[0])

                labels[current_file] = current_label

        data_type = 'validation'

    return partition, labels


def save_model(model, name):
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

# ______________________________________________________________________________________________________________________


# Parameters
params = {'dim': (sample_rate, 1),
          'batch_size': batch_size,
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
tensorboard = TensorBoard(log_dir="log_data/{}".format(time()))

history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=False,
                              epochs=epochs,
                              callbacks=[early_stopping_monitor, tensorboard])

save_model(model, "synthetic_note_trained")

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

