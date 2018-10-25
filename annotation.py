
import tensorflow as tf

# Helper libraries
import numpy as np
import scipy as sp
from sklearn import preprocessing
import os
import random

# audio libraries
from scipy.io import wavfile

list_of_files = os.listdir("./test_training_set/")
num_files = list_of_files.__len__()
number_of_classes = 88
training_data = np.zeros((num_files, 16384), dtype=float)
labels = np.zeros((num_files, 1), dtype=int)

# annotate train_labels / write train_audio
for i in range(0, num_files):
    path = "./test_training_set/" + list_of_files[i]
    temp_data = wavfile.read(path)
    training_data[i, :] = temp_data[1]
    print(training_data[i,:])
    for rootnote in range(0, number_of_classes + 1):
        if list_of_files[i].startswith("sinus" + str(rootnote) + "_"):
            labels[i, :] = rootnote

import pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()
pickle_out = open("Y.pickle", "wb")
pickle.dump(labels, pickle_out)
pickle_out.close()