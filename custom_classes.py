import numpy as np
import keras
import librosa.core
import librosa.util

class DataGenerator(keras.utils.Sequence):
    """Generates data for a Keras model"""
    def __init__(self, list_ids, labels, data_type, batch_size=128, dim=(22050, 1), n_channels=1,
                 n_classes=127, shuffle=True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_type = data_type
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        training_data, target = self.__data_generation(list_ids_temp)

        return training_data, target

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples""" # X : (n_samples, *dim, n_channels)
        # Initialization
        # maybe channels adds additional dimension
        x_data = np.empty((self.batch_size, *self.dim))
        y_data = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, id in enumerate(list_ids_temp):
            # Store sample
            if self.data_type == 'training':
                path = 'E:/datasets/synthetic_note_dataset/training/' + id
            if self.data_type == 'validation':
                path = 'E:/datasets/synthetic_note_dataset/validation/' + id

            audio_data = librosa.core.load(path)
            audio_data = audio_data[0]
            audio_data = librosa.util.normalize(audio_data)
            audio_data = audio_data[:, np.newaxis]
            x_data[i, :, ] = audio_data[0:self.dim[0]]

            # Store class
            y_data[i] = self.labels[id]

        return x_data, keras.utils.to_categorical(y_data, num_classes=self.n_classes)
