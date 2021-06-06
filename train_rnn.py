import os
from train import CVAE_FULL
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import tqdm
import gait
from tqdm import tqdm
import cv2

OneRNN = False

image_vec_shape = 12
gait_key_poses = 16
conditional_vec_shape = 4
pose_image_shape = (160,160,1)
batch_size = 50
full_vec_shape = image_vec_shape+conditional_vec_shape
n_timesteps = 3

model_OneRNN = keras.models.Sequential()
model_OneRNN.add(keras.layers.LSTM(256,
                                   activation='tanh',
                                   return_sequences=True,
                                   input_shape=(n_timesteps,image_vec_shape+conditional_vec_shape)
                                  ))
model_OneRNN.add(keras.layers.LSTM(128,
                                   activation='tanh',
                                   return_sequences=True
                                  ))
model_OneRNN.add(keras.layers.LSTM(64,
                                   activation='tanh'
                                  ))
model_OneRNN.add(keras.layers.Dense(image_vec_shape+conditional_vec_shape,
                                    activation='linear'
                                  ))
model_OneRNN.compile(optimizer='adam', loss='mse')

def main():
    if os.path.isfile(os.getcwd()+'/weights/CVAE_FULL.h5'):
        CVAE_FULL.load_weights('weights/CVAE_FULL.h5')
        print("Loaded weights successfully")
    else:
        raise Exception("No Weights found !!")

    encoder = keras.models.Model(CVAE_FULL.input, CVAE_FULL.get_layer("concat_zcond").output)

    encoded_vec = gait.encode_data(encoder,label_angle='090')

    x, y = gait.encoded2timeseries(encoded_vec,3)
    if os.path.isfile(os.getcwd()+'/weights/OneRNN.h5'):
        model_OneRNN.load_weights('weights/OneRNN.h5')
    history = model_OneRNN.fit(x, y, epochs=250, validation_split=0.1, batch_size=64)
    return history

if __name__ == "__main__":
    main()