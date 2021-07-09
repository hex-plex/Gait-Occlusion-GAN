import keras
import numpy as np


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
