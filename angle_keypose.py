import re
import os
import cv2
from keypose_paths import get_keyposepath

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
    # paths_k = get_keyposepath(cluster = keypose, dir=data_path)
    # paths_k.sort()

    # if not(type(paths_k)==list):
    #     print('Error, \'paths_k\' type: not got a list')
    #     return

    # re_mask = re.compile('.*/'+angle+'($|/.*)')
    # results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
    # images = [cv2.imread(file) for file in results]

    subject = sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh/'+'001'))
    images = []
    for sub in subject:

        re_mask = re.compile('.*/'+exp+'/'+sub+'/'+ang+'($|/.*)')
        results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
        
        if len(results)>2:
            # ims = np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))])
            ds.append(np.asarray([cv2.imread(file) for file,_ in zip(results,range(3))]))

    return images