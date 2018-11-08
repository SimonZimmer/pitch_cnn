
import tensorflow as tf

# Helper libraries
import numpy as np
import os
import pickle
import keras.utils

# audio libraries
from scipy.io import wavfile

list_of_files = os.listdir("./test_training_set/")
num_files = list_of_files.__len__()
num_classes = 88
training_data = np.zeros((num_files, 16384), dtype=float)
labels = np.zeros((num_files, 1), dtype=int)

# annotate train_labels / write train_audio
for i in range(0, num_files):
    path = "./test_training_set/" + list_of_files[i]
    temp_data = wavfile.read(path)
    training_data[i, :] = temp_data[1]
    print(training_data[i, :])
    for rootnote in range(0, num_classes + 1):
        if list_of_files[i].startswith("sinus" + str(rootnote) + "_"):
            labels[i, :] = rootnote

# convert to one-hot matrix
labels = keras.utils.to_categorical(labels, num_classes)

# save np arrays using pickle
pickle_out = open("x_train.pickle", "wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()
pickle_out = open("y_train.pickle", "wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

# pickle_out = open("y_validate.pickle", "wb")
# pickle.dump(labels[9 * int(labels.shape[0]/10):labels.shape[0]], pickle_out)
# pickle_out.close()
# pickle_out = open("x_validate.pickle", "wb")
# pickle.dump(training_data[9 * int(training_data.shape[0]/10):training_data.shape[0]], pickle_out)
# pickle_out.close()
