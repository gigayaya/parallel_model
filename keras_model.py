from keras.models import *
from keras.layers import *
from keras.optimizers import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def get_model():
    model = Sequential()
    model.add(Dense(units=10, input_dim=1,activation='relu'))
    model.add(Dense(units=1,activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    pass