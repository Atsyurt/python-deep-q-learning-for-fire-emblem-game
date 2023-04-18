# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 09:40:13 2023

@author: ayhant
"""


from matplotlib import image
from matplotlib import pyplot
import numpy as np
import random
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from DDQN_Agent import DDQN_Agent

available_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ddqn_agent=DDQN_Agent((640,640,3),len(available_moves))

# load image as pixel array
image = image.imread('./screenshot.png')
data = np.asarray(image)
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()





avg_image = np.expand_dims(np.zeros((data.shape), np.float64), axis=0)
avg_image += data


def  blend_images (images, blend):
    avg_image = np.expand_dims(np.zeros((88, 80, 1), np.float64), axis=0)

    for image in images:
        avg_image += image
        
    if len(images) < blend:
        return avg_image / len(images)
    else:
        return avg_image / blend

def process_frame(frame):

    data_img = np.array([210, 164, 74]).mean()
    # img = frame[::2, ::2]    # Crop and downsize
    img = (frame - 128) / 128 - 1  # Normalize from -1 to 1.
    
    return np.expand_dims(img, axis=0)

input_img=process_frame(data)
ddqn_agent.act(input_img)

inputs = tf.keras.Input(shape=(input_img.shape[1],input_img.shape[2],input_img.shape[3]))
c1 = tf.keras.layers.Conv2D(32, (8, 8), strides=2, padding='same',)(inputs)
c1a=tf.keras.layers.Activation(activations.relu)(c1)


c2 = tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same')(c1a)
c2a=tf.keras.layers.Activation(activations.relu)(c2)


c3 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same')(c2a)
c3a=tf.keras.layers.Activation(activations.relu)(c3)


flatten_layer=tf.keras.layers.Flatten()(c3a)

d1=tf.keras.layers.Dense(128)(flatten_layer)
d1a=tf.keras.layers.Activation(activations.relu)(d1)


d3=tf.keras.layers.Dense(64)(d1a)
d3a=tf.keras.layers.Activation(activations.relu)(d3)

d4=tf.keras.layers.Dense(5)(d3a)
final_activation=tf.keras.layers.Activation(activations.linear)(d4)


model = tf.keras.Model(inputs=inputs, outputs=final_activation)
model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=None, decay=0.0))

model.summary()

act_values=model.predict(input_img)
np.argmax(act_values[0])

model.summary()





# model = Sequential()

# model.add(keras.Input(shape=data.shape))
# # Conv Layers
# model.add(Conv2D(32, (8, 8), strides=2, padding='same'))
# model.add(Activation('relu'))

# model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
# model.add(Activation('relu'))

# model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
# model.add(Activation('relu'))
# model.add(Flatten())

# # FC Layers
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(5, activation='linear'))

# model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=None, decay=0.0))


