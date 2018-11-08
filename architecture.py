
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt

#load data
pickle_in = open("x_train.pickle", "rb")
x_train = pickle.load(pickle_in)
pickle_in = open("y_train.pickle", "rb")
y_train= pickle.load(pickle_in)

# normalize data
x_train = np.abs(x_train / np.max(x_train))
x_train = np.expand_dims(x_train, axis=2)

print("size of x_train", x_train.shape)
print("size of y_train", y_train.shape)


number = x_train.shape[0]
size = x_train.shape[1]
sample_rate = pow(2, 14)
kernel_size = np.int(np.floor(sample_rate / 100))

epochs = 10;
batch_size = 128;
num_classes = 88;


# build model architecture
model = keras.Sequential()

model.add(keras.layers.Conv1D(input_shape=(size, 1),
                              filters=48,
                              kernel_size=kernel_size,
                              strides=4))

model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(filters=48,
                              kernel_size=3,
                              strides=1,))

model.add(keras.layers.Conv1D(filters=48,
                              kernel_size=3,
                              strides=1,))

model.add(keras.layers.Conv1D(filters=48,
                              kernel_size=3,
                              strides=1,))

model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(filters=96,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=96,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=96,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=96,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(filters=192,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=192,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=192,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=192,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=192,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=192,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.MaxPooling1D(pool_size=4))

model.add(keras.layers.Conv1D(filters=384,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=384,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.Conv1D(filters=384,
                              kernel_size=3,
                              strides=1))

model.add(keras.layers.AveragePooling1D())

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(num_classes, activation='softmax'))


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model
history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# plot results
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('accuray')
plt.ylabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('loss')
plt.ylabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()