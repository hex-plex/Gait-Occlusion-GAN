import re
import os
import cv2
from keypose_paths import get_keyposepath

def angle_ims(angle=0,keypose = 4,data_path='/home/ishikaa/Downloads'):
    '''
    Returns images of certain angle for a keypose.

    angle: Angle
    keypose_paths: list of paths of images that belong to specifc keypose
    '''

    os.chdir(data_path)

    angle = f"{angle:03}"
    paths_k = get_keyposepath(cluster = keypose, dir=data_path)
    paths_k.sort()

    if not(type(paths_k)==list):
        print('Error, \'paths_k\' type: not got a list')
        return

    re_mask = re.compile('.*/'+angle+'($|/.*)')
    results = [re_mask.search(str(path)).group(0) for path in paths_k if re_mask.search(str(path))]
    images = [cv2.imread(file) for file in results]

    return images