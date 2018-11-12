
# Helper libraries
import numpy as np
import os

# audio libraries
import scipy.io.wavfile

list_of_files = os.listdir("./main_training_set/")
num_files = list_of_files.__len__()
start_note = 28
end_note = 65
num_classes = end_note - start_note + 1

training_data = np.zeros((num_files, 16384), dtype=float)
labels = np.zeros((num_files, 1), dtype=int)

print("annotating...")
for i in range(num_files):

    if list_of_files[i] == '.DS_Store':
        print('DS_Store item encountered & removed')
        os.remove(list_of_files[i])

    path = "./main_training_set/" + list_of_files[i]
    temp_data = scipy.io.wavfile.read(path)
    training_data[i, :] = temp_data[1]

    for rootnote in range(num_classes):
        if list_of_files[i].startswith("sinus" + str(rootnote + start_note) + "_"):
            labels[i] = rootnote

print("writing data to disk...")
np.save("/Users/simonzimmermann/dev/pitch_cnn/x_train.npy", training_data)
np.save("/Users/simonzimmermann/dev/pitch_cnn/y_train.npy", labels)
