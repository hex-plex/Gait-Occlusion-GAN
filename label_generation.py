import numpy as np
from gait import *
import cv2
import os
import pickle

def gen():
    os.chdir('/home/ishikaa/Downloads/')
    kmeans = kmean_train(subject='001',choice='bg-01',override=True)
    ret = supervision(kmeans,override=True)
    # if ret:
    #     a = fetch_labels()