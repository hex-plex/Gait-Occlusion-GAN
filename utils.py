import re
import os
import cv2
import numpy as np
from pathlib import Path
import pickle
from torch.cuda import is_available

def get_keyposepath(cluster=4,dir='/home/ishikaa/Downloads'):
    '''
    Returns paths to all images belonging to certain keypose
    '''

    clusters = []
    paths = []
    for path in Path(dir).rglob('labels.pkl'):
        file_to_read = open(path, "rb")
        temp_dict = pickle.load(file_to_read)
        temp_cluster = [k for k,v in temp_dict.items() if v==cluster]
        temp_path = [path for _,v in temp_dict.items() if v==cluster]
        clusters = clusters + temp_cluster
        paths = paths + temp_path

    paths_k = [str(path).replace('labels.pkl',cluster) for path,cluster in zip(paths,clusters)]

    return paths_k

def angle_ims(exp=1,angle=0,keypose = 4,data_path='/home/ishikaa/Downloads'):
    '''
    Returns images of certain angle for a keypose. (IN A GIVEN EXPERIMENT)

    Args:
    exp: Experiment
    angle: Angle
    keypose_paths: list of paths of images that belong to specifc keypose
    '''

    os.chdir(data_path)
    exp = f"{exp:03}"
    angle = f"{angle:03}"
    paths_k = get_keyposepath(cluster = keypose, dir=data_path)
    paths_k.sort()

    # if not(type(paths_k)==list):
    #     print('Error, \'paths_k\' type: not got a list')
    #     return

    # re_mask = re.compile('.*/'+angle+'($|/.*)')
    # results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
    # images = [cv2.imread(file) for file in results]

    subject = sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh/'+exp))
    images = []
    for sub in subject:

        re_mask = re.compile('.*/'+exp+'/'+sub+'/'+angle+'($|/.*)')
        results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
        
        if len(results)>2:
            # ims = np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))])
            images.append(np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))]))

    return images

def get_device():
    if is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
