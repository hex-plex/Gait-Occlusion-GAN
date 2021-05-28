from gait import preprocess
import os
import numpy as np
import cv2



def make_dataset():
    os.chdir('/home/ishikaa/Downloads/')
    subjects = []
    x = []

    for subfolder in sorted(os.listdir( '/'.join([os.getcwd(),'GaitDatasetB-silh','001']))):
        subject = []
        for file in sorted(os.listdir( '/'.join([os.getcwd(),'GaitDatasetB-silh', '001',subfolder,'090']))):
            subject.append(preprocess(cv2.imread('/'.join([os.getcwd(),'GaitDatasetB-silh/','001',subfolder,'090',file]))))
        
        subjects.append(np.array(subject))

    return subjects
