# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os

# audio libraries
from scipy.io import wavfile

list_of_files = os.listdir("./training_set/")
train_audio = np.zeros((16384, list_of_files.__len__()))
train_labels = np.zeros((1, list_of_files.__len__()))
number_of_classes = 88

# annotate train_labels / write train_audio
i = 0
#for file in list_of_files:
 #   path = "./training_set/" + file
  #  data = wavfile.read(path)
#    train_audio[:, i] = data[1]
   # for rootnote in range(0, number_of_classes + 1):
    #    if file.startswith("sinus" + str(rootnote) + "_"):
     #       train_labels[:, i] = rootnote
   # i = i + 1

np.save("/Users/simonzimmermann/dev/pitch_cnn/train_audio.npy", train_audio)
np.save("/Users/simonzimmermann/dev/pitch_cnn/train_labels.npy", train_labels)

# TODO add class name translation here
# TODO split in train and test here