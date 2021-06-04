import torch.nn as nn
from gait import preprocess
from utils import angle_ims
import numpy as np

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
        l2 = nn.ReLU()
        
        if last:
            return nn.Sequential(l1,l2)
        else:
            l3 = nn.MaxPool3d((1, 1, 1))
            l4 = nn.BatchNorm3d(out_channels)
            return nn.Sequential(l1,l2,l3,l4)
        
        
    def forward(self,x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        return out

class PEIData(Dataset):

    def __init__(self, num_exps,transform=None):
        """
        Custom dataset for images of a certain keypose at a given angle.

        Args:
            angle (int)     : Angle
            keypose (int)   : Key-pose/Cluster
            data_path (str) : Path where dataset is downloaded
        """
        
        ds = []
        for i in range(num_exps):
            exp = angle_ims(exp=i+1,angle=0,keypose = 4)
            ds = ds + exp

        images = np.empty((len(ds),3,ds[0].shape[2]//2,ds[0].shape[1]//2))

        for i in range(len(ds)):
            images[i] = np.asarray([preprocess(im)/255 for im in ds[i]])
        
        self.images = images.reshape(images.shape[0],1,images.shape[1],images.shape[2],images.shape[3]).astype('float32')

        #Avg PEI after PCA .
        # self.y = np.mean([image for image in images_0_4],axis=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


