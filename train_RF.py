import os
from train import CVAE_FULL
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import gait
from tqdm import tqdm
import cv2
from train_birnn import model_BiRNN, ImageNoise
from sklearn.ensemble import RandomForestClassifier 
import gc

labels = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06']


pr_imgs = [None for _ in labels]
names = [None for _ in labels]
pro_imgs = [None for _ in labels]


for i, label in enumerate(labels):
    _, pr_imgs[i], names[i] = gait.fetch_data(label,90,True, True, True)
    
    
key_pose_info = gait.fetch_labels()
    
gc.collect()

y = []
for i in range(len(labels)):
    pro_imgs[i] = np.moveaxis(np.concatenate(pr_imgs[i], axis=-1), -1, 0) 
    names[i] = np.concatenate(names[i], axis=-1)
    y.append(i*np.ones((len(pr_imgs[i],))))
datasetImgs = np.concatenate(pro_imgs, axis=0)
    
datasetImgs = np.concatenate(pr_imgs, axis=0)

X, _, _, _, _, info = gait.get_feature_vectors(datasetImgs,preproc=lambda x:x)

eigvec, eigvalue = info['eigvec'], info['eigvalue']
np.savez_compressed('eig90',eigvec, eigvalue)