import numpy as np
from gait_osaka import *
import cv2
import os
import pickle

kmeans = kmean_train(subject='00111',override=True)
ret = supervision(kmeans,override=True)
if ret:
    a = fetch_labels(save=True,override=True)
