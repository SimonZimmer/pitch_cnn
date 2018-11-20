
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorboard
import numpy as np
import keras
import os
import scipy.io.wavfile
from sklearn.model_selection import train_test_split

start_note = 28
end_note = 65
num_classes = end_note - start_note + 1
sample_rate = pow(2, 14)
init_kernel_size = np.int(np.floor(sample_rate / 100))
epochs = 100
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
    #x = residual_block(x, 48, 48)
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
    # x = residual_block(x, 192, 192)
    # x = residual_block(x, 192, 192)
    # x = residual_block(x, 192, 192)
    # x = keras.layers.MaxPool1D(pool_size=4)(x)

    # conv5
    for i in range(3):
        x = keras.layers.Conv1D(filters=384,
                                kernel_size=3,
                                strides=1,
                                padding="same")(x)
        x = batch_and_relu(x)
    # x = residual_block(x, 192, 384)
    # x = keras.layers.Conv1D(filters=48,
    # kernel_size=3,
    # strides=1,
    # padding="same")(x)
    # x = batch_and_relu(x)

    x = keras.layers.AveragePooling1D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=audio, outputs=x)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    return model


def train(model, x_train, y_train, epochs, with_plot):
    early_stopping_monitor = keras.callbacks.EarlyStopping(patience=2)

    tensorboard("logs/run_a")

    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        batch_size=128,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[early_stopping_monitor, keras.callbacks.TensorBoard("logs/run_without_res")])

    if with_plot:
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.xlabel('accuray')
        plt.ylabel('epochs')
        plt.show()

        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.xlabel('loss')
        plt.ylabel('epochs')
        plt.show()

    return history


def load_data(num_classes):
    print("load data...")
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    return x_train, y_train, x_test, y_test


def annotate():
    print("annotate data...")

    seed = 7
    np.random.seed(seed)

    list_of_files = os.listdir("./main_training_set/")
    num_files = list_of_files.__len__()

    training_data = np.zeros((num_files, 16384), dtype=float)
    labels = np.zeros((num_files, 1), dtype=int)

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

    x_train, x_test, y_train, y_test = train_test_split(training_data,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=seed)
    print("writing data to disk...")
    np.save("/Users/simonzimmermann/dev/pitch_cnn/x_train.npy", x_train)
    np.save("/Users/simonzimmermann/dev/pitch_cnn/y_train.npy", y_train)
    np.save("/Users/simonzimmermann/dev/pitch_cnn/x_test.npy", x_test)
    np.save("/Users/simonzimmermann/dev/pitch_cnn/y_test.npy", y_test)


def save_model(name):
    model_json = model.to_json()
    with open("%s.json" % name, "w") as json_file:
        json_file.write(model_json)

    model.save_weights("%s.h5" % name)
    print("Saved model to disk")


def load_model(name):
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tf.keras.models.model_from_json(
        loaded_model_json
    )

    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    return loaded_model


def predict(model, x_test):
    y = model.predict(x_test)
    indices = np.argmax(y, axis=1)

    return indices


def evaluate(model):
    model.compile(loss='binary_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])
    cvscores = []
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)


# execution:
x_train, y_train, x_test, y_test = load_data(num_classes)
size = x_train.shape[1]
model = build_model(size)
model.summary()
train(model, x_train, y_train, epochs, True)





