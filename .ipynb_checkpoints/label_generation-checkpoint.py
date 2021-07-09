import numpy as np
from gait_tumiitkgp import *
import cv2
import os
import pickle

kmeans = kmean_train(subject='static_occ_id024_6',override=True)
ret = supervision(kmeans,override=True)
if ret:
    a = fetch_labels()
