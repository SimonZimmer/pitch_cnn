# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import preprocessing
import os

# audio libraries
from scipy.io import wavfile

list_of_files = os.listdir("./test_training_set/")
num_files = list_of_files.__len__()
number_of_classes = 88
audio_data = np.zeros((num_files, 16384), dtype=float)
labels = np.zeros((num_files, 1), int)

# annotate train_labels / write train_audio

for i in range(0, num_files):
    path = "./test_training_set/" + list_of_files[i]
    temp_data = wavfile.read(path)
    temp_data = np.abs(temp_data[1] / temp_data[1].max())
    audio_data[i, :] = temp_data[1]
    for rootnote in range(0, number_of_classes + 1):
        if list_of_files[i].startswith("sinus" + str(rootnote) + "_"):
            labels[i, :] = rootnote
# check if numbers match
assert audio_data.shape[0] == labels.shape[0]

# create tf file
audio_data_placeholder = tf.placeholder(audio_data.dtype, audio_data.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((audio_data_placeholder, labels_placeholder))

# build iterator for dataset
iterator = dataset.make_initializable_iterator()
sess = tf.Session()
sess.run(iterator.initializer, feed_dict={audio_data_placeholder: audio_data,
                                          labels_placeholder: labels})