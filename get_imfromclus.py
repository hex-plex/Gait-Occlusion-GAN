
from pathlib import Path
import pickle

def getims(cluster=k,path='/home/ishikaa/Downloads'):
    clusters = []
    paths = []

    for path in Path(path).rglob('labels.pkl'):
        file_to_read = open(path, "rb")
        temp_dict = pickle.load(file_to_read)
        temp_cluster = [k for k,v in temp_dict.items() if v==4]
        temp_path = [path for _,v in temp_dict.items() if v==4]
        clusters = clusters + temp_cluster
        paths = paths + temp_path

    paths_k = [str(path).replace('labels.pkl',cluster) for path,cluster in zip(paths4,cluster4)]

    return paths_k