from keras.models import Sequential
import matplotlib as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, Lambda, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
def make_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 1.,input_shape=(64, 64, 3),output_shape=(64, 64, 3)))
    model.add(Conv2D(filters=16,kernel_size=(3,3), input_shape=(64,64,3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(filters=32,kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64,kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(filters=128,kernel_size=(3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(output_dim=512))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_dim=256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(output_dim=2))
    model.add(Activation("softmax"))
    return model
