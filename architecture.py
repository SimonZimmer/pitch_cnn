
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

epochs = 10
batch_size = 128
start_note = 28
end_note = 65
num_classes = end_note - start_note + 1

print("load data...")
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
y_train = keras.utils.to_categorical(y_train, num_classes)
x_train = np.expand_dims(x_train, axis=2)
print("size of x_train", x_train.shape)
print("size of y_train", y_train.shape)

number = x_train.shape[0]
size = x_train.shape[1]
sample_rate = pow(2, 14)
kernel_size = np.int(np.floor(sample_rate / 100))

print("building model architecture...")
model = keras.Sequential()


def res_layer_pair(x, filters, kernel_size, strides, pool_size, pool=False):
    res = x
    if pool:
        x = keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        res = keras.layers.Conv1D(filters=filters,
                                  kernel_size=1,
                                  strides=2)(res)

    out = keras.layers.BatchNormalization()(x)
    out = keras.layers.Activation("relu")(out)
    out = keras.layers.Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              strides=strides)(out)

    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.Activation("relu")(out)
    out = keras.layers.Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              strides=strides)(out)

    keras.layers.add([res, out])

    return out

model.add(keras.layers.Conv1D(input_shape=(size, 1),
                              filters=48,
                              kernel_size=kernel_size,
                              strides=4))



model.add(keras.layers.MaxPooling1D(pool_size=2))

for i in range(3):
    model.add(keras.layers.Conv1D(filters=48,
                                  kernel_size=3,
                                  strides=1))

model.add(keras.layers.MaxPooling1D(pool_size=4))

for i in range(4):
    model.add(keras.layers.Conv1D(filters=96,
                                  kernel_size=3,
                                  strides=1))

model.add(keras.layers.MaxPooling1D(pool_size=4))

for i in range(6):
    model.add(keras.layers.Conv1D(filters=192,
                                  kernel_size=3,
                                  strides=1))

model.add(keras.layers.MaxPooling1D(pool_size=4))

for i in range(3):
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

#TODO add activation function after each layer (relu)
#TODO add batch normalisation after each activation function