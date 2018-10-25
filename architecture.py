
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Helper libraries
import numpy as np
import pickle

#load data
pickle_in = open("X.pickle", "rb")
x = pickle.load(pickle_in)

pickle_in = open("Y.pickle", "rb")
y = pickle.load(pickle_in)

# normalize
x = np.abs(x / np.max(x))

model = Sequential()
model.add(Conv2D((32000, 1), (4, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4, 1)))

model.add(Flatten())

#model.add(Dense(1))
#model.add(Activation("sigmoid"))

model.compile(loss="crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x, y, batch_size=32, validation_split=0.1)





