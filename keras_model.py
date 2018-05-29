from keras.models import *
from keras.layers import *
from keras.optimizers import *

def get_model():
    model = Sequential()
    model.add(Dense(units=10, input_dim=1,activation='relu'))
    model.add(Dense(units=1,activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    pass