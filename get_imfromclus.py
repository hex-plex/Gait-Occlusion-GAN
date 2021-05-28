
from pathlib import Path
import pickle

def getims(cluster=4,dir='/home/ishikaa/Downloads'):
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