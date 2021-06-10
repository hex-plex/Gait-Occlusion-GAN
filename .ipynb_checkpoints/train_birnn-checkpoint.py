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

model_BiRNN = keras.models.Sequential()
model_BiRNN.add(keras.layers.Bidirectional(
                    keras.layers.LSTM(256,
                                      activation='tanh',
                                      return_sequences=True),
                                      input_shape=(2*n_timesteps,image_vec_shape+conditional_vec_shape)
                ))
model_BiRNN.add(keras.layers.Bidirectional(
                    keras.layers.LSTM(128,
                                      activation='tanh',
                                      return_sequences=True)
                ))
model_BiRNN.add(keras.layers.LSTM(64,
                                  activation='tanh',
                                  return_sequences=True
                                 ))
model_BiRNN.add(keras.layers.TimeDistributed(
                    keras.layers.Dense(image_vec_shape+conditional_vec_shape)
                ))
model_BiRNN.compile(optimizer='adam', loss='mse')


class ImageNoise(keras.utils.Sequence):
    def __init__(self, encodedVec, processTimeseries, timesteps=6, batch_size = 50, n_classes=16, occlusionRatio=0.08, shuffle=True):
        self.encodedData ,_ = processTimeseries(encodedVec, timesteps, y_out = False)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.occlusionRatio = occlusionRatio
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.encodedData)/self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = self.addNoise(self.encodedData[indexes].copy(), self.occlusionRatio)
        Y = self.encodedData[indexes].copy()
        
        return X,Y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.encodedData))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def addNoise(self, arr, ratio=0.07):
        """
            ratio percent of images are occluded
        """
        return (np.random.random(arr.shape[:2])>ratio).reshape(*arr.shape[:2],1)*arr


def main():
    if os.path.isfile(os.getcwd()+'/weights/CVAE_FULL.h5'):
        CVAE_FULL.load_weights('weights/CVAE_FULL.h5')
        print("Loaded weights successfully")
    else:
        raise Exception("No Weights found !!")

    encoder = keras.models.Model(CVAE_FULL.input, CVAE_FULL.get_layer("concat_zcond").output)

    encoded_vec = gait.encode_data(encoder,label_angle='090')

    
    if os.path.isfile(os.getcwd()+'/weights/BiRNN.h5'):
        model_BiRNN.load_weights('weights/BiRNN.h5')
        
    train_vec = { ele:encoded_vec[ele] for ele in list(encoded_vec.keys())[:-100] }
    valid_vec = { ele:encoded_vec[ele] for ele in list(encoded_vec.keys())[-100:] }

    train_data = ImageNoise(train_vec,gait.encoded2timeseries,occlusionRatio=1./6)
    valid_data = ImageNoise(valid_vec,gait.encoded2timeseries,occlusionRatio=1./6)

    history = model_BiRNN.fit(train_data, 
                              validation_data = valid_data, 
                              epochs=1000,
                              use_multiprocessing=True,
                              workers=4)
    
    model_BiRNN.save("weights/BiRNN.h5")
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("BiRNN_training.jpg")
    return history

if __name__ == "__main__":
    main()