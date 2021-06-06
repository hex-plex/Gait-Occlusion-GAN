import re
import os
import cv2
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from gait import preprocess,fetch_labels

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

    subject = sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh/'+exp))
    images = []
    paths = []
    for sub in subject:

        re_mask = re.compile('.*/'+exp+'/'+sub+'/'+angle+'($|/.*)')
        results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
        
        if len(results)>2:
            # ims = np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))])
            # images.append(np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))]))
            paths.append([file for file,_ in results])

    return paths



def imshow(model_out,actual):
    # img = img / 2 + 0.5  
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,4))
    axes[0].imshow(model_out,cmap='gray',vmin=0,vmax=1)
    axes[0].set_title('Model')
    axes[1].imshow(actual,cmap='gray',vmin=0,vmax=1)
    axes[1].set_title('Average')
    plt.show()

def ims(exp=1,angle=0,paths_k=[],data_path='/home/ishikaa/Downloads'):
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


    subject = sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh/'+exp))
    # images = []
    paths = []
    for sub in subject:

        re_mask = re.compile('.*/'+exp+'/'+sub+'/'+angle+'($|/.*)')
        results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
        
        if len(results)>2:
            # ims = np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))])
            # images.append(np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))]))
            paths.append([file for file,_ in zip(results,range(3))])

    return paths

def check(exp=1,angle=0,paths_k=[],data_path='/home/ishikaa/Downloads'):
    '''
    FAKE
    Returns images of certain angle for a keypose. (IN A GIVEN EXPERIMENT)

    Args:
    exp: Experiment
    angle: Angle
    keypose_paths: list of paths of images that belong to specifc keypose
    '''

    os.chdir(data_path)
    exp = f"{exp:03}"
    angle = f"{angle:03}"


    subject = sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh/'+exp))
    # images = []
    paths = []
    for sub in subject:

        re_mask = re.compile('.*/'+exp+'/'+sub+'/'+angle+'($|/.*)')
        results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
        
        if len(results)>2:
            # ims = np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))])
            # images.append(np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))]))
            paths.append([file for file in results])

    return paths
