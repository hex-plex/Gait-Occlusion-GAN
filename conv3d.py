import torch.nn as nn
from torch.utils.data import Dataset
from gait import preprocess
from utils import check,get_keyposepath,ims
import numpy as np
import cv2

class Conv3D(nn.Module):

    '''
    Performs 3D convolution
    '''

    def __init__(self):

        super(Conv3D, self).__init__()
        self.conv1 = self._convblock(1,16,3,1,False)
        self.conv2 = self._convblock(16,32,1,1,False)
        self.conv3 = self._convblock(32,16,1,1,False)
        self.conv4 = self._convblock(16,1,1,1,True)

    def _convblock(self,in_channels,out_channels,ksized,ksize,last):
        '''
        Makes a block of layers (Conv3d,ReLU,Maxpool3d,BatchNorm3d(only if !last))
        '''

        l1 = nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=(ksized,ksize,ksize))
        l2 = nn.BatchNorm3d(out_channels)
        l3 = nn.MaxPool3d((1, 1, 1))
        l4 = nn.ReLU()

        
        if last:
            return nn.Sequential(l1,l2,l4)
        else:
            
            return nn.Sequential(l1,l2,l3,l4)
        
        
    def forward(self,x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        return out

class PEI(Dataset):

    def __init__(self, num_exps=1,keyposes=[4], angle=0):
        """
        Custom dataset for images of a certain keypose at a given angle.

        Args:
            angle (int)     : Angle
            keypose (list)  : Key-poses/Clusters
            data_path (str) : Path where dataset is downloaded
        """
        paths_k = [sorted(get_keyposepath(cluster = i)) for i in keyposes]

        self.ds = [] #Paths to all images d[0]=> subject 1  (len = num of frames for it)
        for i in range(num_exps):
            for j in range(len(keyposes)):
                exp = check(exp=i+1,angle=angle,paths_k=paths_k[j])
                self.ds = self.ds + exp

        new = []
        num = 3
        for sub in self.ds:
            if len(sub)>num:
                temp = []
                for i in range(len(sub)-num+1):
                    new.append(sub[i:i+num])
            else:
                new.append(sub)
        
        self.ds = new
        

    def __len__(self):

        return len(self.ds)

    def __getitem__(self, idx):

        frames = np.asarray([preprocess(cv2.imread(im))/255. for im in self.ds[idx]]) #Shape: (3,160,120)
        y = np.mean([image for image in frames],axis=0) #Shape (160,120)
        mask = np.random.random(frames.shape[0])<0.5
        frames[mask] = np.zeros((frames.shape[1],frames.shape[2]))
        # X = self.occlude()
        
        
        return frames.reshape(1,frames.shape[0],frames.shape[1],frames.shape[2]).astype('float32'),y.astype('float32')
