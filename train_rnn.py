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

if False and os.path.isfile(os.getcwd()+'/weights/CVAE_FULL.h5'):
    CVAE_FULL.load_weights('weights/CVAE_FULL.h5')
    print("Loaded weights successfully")
elif False:
    raise Exception("No Weights found !!")

encoder = keras.models.Model(CVAE_FULL.input, CVAE_FULL.get_layer("concat_zcond").output)

labels = gait.fetch_labels(label_angle="090",save=False,override=True)

files = [filename for filename in labels]

imgs = np.empty((50,160,160,1))
z = np.empty((50),dtype=int)
for i,file in enumerate(files[0:50]):
    imgs[i,] = cv2.copyMakeBorder(gait.preprocess(cv2.imread(file)), 0, 0, 20, 20, cv2.BORDER_CONSTANT, (0,0,0)).reshape(160,160,1)/255.
    z[i] = labels[file]
z_vec = keras.utils.to_categorical(z, num_classes=16)

enc_vec = encoder.predict([imgs,z_vec],batch_size=50)
