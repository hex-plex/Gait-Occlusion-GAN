import cv2
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
import keras

def preprocess(img):
    cnt, heir = cv2.findContours(img[:,:,0],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnt)>0):
        x,y,w,h = cv2.boundingRect(cnt[0])
        temp = img[y:y+h, x:x+w, 0]
        w1 = (h*3)//4
        temp = cv2.copyMakeBorder(temp, 0,0, max(w1-w,0)//2, max(w1-w,0)//2, cv2.BORDER_CONSTANT, (0,0,0))
        return cv2.resize(temp, (120,160))
    else:
        return np.zeros((160,120))

def get_feature_vectors(imgs,k=10):
    G = np.vstack(tuple(preprocess(img).reshape(-1).astype(np.float64)/255. for img in imgs))
    avg = G.T.mean(axis=1)
    A = G.T-avg.reshape(-1,1)
    X = A.T@A / A.shape[1]
    eigvalue, eigvec = np.linalg.eigh(X)
    U = A@eigvec
    u = U/np.linalg.norm(U, axis=0)
    u_sorted = u[:,np.argsort(eigvalue)[::-1]]
    u_k = u_sorted[:,0:k]
    W = u_k.T@A
    A1 = u_k@W
    return W, A1, A, u_k, avg

class KMeans():
    def __init__(self, K = 16, debug=False):
        self.K = K
        self.debug = debug
        self.states = [[] for _ in range(self.K)]
        self.P = np.zeros((10,self.K))
    def fit(self,W):
        self.W = W
        for j in range(self.W.shape[1]):
            self.states[int(j*self.K/self.W.shape[1])].append(j)
        for i, state in enumerate(self.states):
            self.P[:,i] = self.W[:,state].mean(axis=1)
        self.converge()
    def converge(self):
        changes = 0
        while changes < 5 :
            prev_states = self.states
            for i, state in enumerate(self.states):
                self.P[:,i] = self.W[:,state].mean(axis=1)
            temp_states = []
            for i in range(self.K):
                temp_state = []
                for j in [(i-1)%self.K, (i+1)%self.K]:
                    if i == j: continue
                    for state in self.states[j]:
                        if np.linalg.norm(self.W[:,state]-self.P[:,i]) <= np.linalg.norm(self.W[:,state]-self.P[:,j]):
                            if self.debug : print(i,j)
                            temp_state.append([j,state])
                temp_states.append(temp_state)
            for i, temp_state in enumerate(temp_states):
                temp_state.sort(key=lambda x : -x[1] if x[0]<i else x[0])
                for clst, frm in temp_state:
                    if clst<i:
                        if self.states[clst][-1] == frm and len(self.states[clst])>1:
                            self.states[clst].remove(frm)
                        else:
                            temp_states[i].remove([clst,frm])
                    else:
                        if self.states[clst][0] == frm and len(self.states[clst])>1:
                            self.states[clst].remove(frm)
                        else:
                            temp_states[i].remove([clst,frm])
            for i in range(len(self.states)):
                for _, x in temp_states[i]:
                    self.states[i].append(x)
                self.states[i].sort()
            if self.states != prev_states:
                changes = 0
            else:
                changes += 1
    def predict(self, W_t):
        MV = np.array(tuple(np.linalg.norm(W_t - self.P[:,i:i+1],axis=0) for i in range(self.K)))
        MV = 1 - MV / MV.max(axis=0)
        return MV

def graph_sort(k_means, W_i,plot=False, A_i=None, avg_i=None):
    if plot:
        if (A_i is None) or (avg_i is None):
            raise Exception('For plot both A_i and avg_i must be passed')
    E = k_means.predict(W_i)
    V = np.zeros((E.shape[0]*E.shape[1], E.shape[0]*E.shape[1]))
    decode = lambda x,k : (x%k,x//k)
    encode = lambda i,j,k: j*k + i
    for cnt in range(V.shape[0]-E.shape[0]):
        i,j = decode(cnt, E.shape[0])
        for pos in [(i,j+1),((i+1)%E.shape[0],j+1)]:
            V[cnt,encode(*pos, E.shape[0])] = 1    # if V[x][y] == 1 then x->y only
    PV = np.zeros_like(E)
    for curr in range(V.shape[0]):
        maxx = 0
        for prev in np.where(V[:,curr]==1)[0]:
            temp = PV[decode(prev,E.shape[0])]
            if temp>maxx:
                maxx = temp
        PV[decode(curr,E.shape[0])] = maxx + E[decode(curr,E.shape[0])]
    term_state = (np.argmax(PV[:,-1]),E.shape[1]-1)
    test_states = []
    test_states.append(term_state[0])
    prevs = np.where(V[:,encode(*term_state,E.shape[0])]==1)[0]
    while len(prevs)>0:
        maxx = 0
        pos = 0
        for prev in prevs:
            if maxx < PV[decode(prev,E.shape[0])]:
                maxx = PV[decode(prev,E.shape[0])]
                pos = prev
        term_state = decode(pos,E.shape[0])
        test_states.append(term_state[0])
        prevs = np.where(V[:,encode(*term_state,E.shape[0])]==1)[0]
    test_states = np.asarray(test_states[::-1])
    if plot:
        cols = 10
        rows = (test_states.shape[0]//cols + (1 if test_states.shape[0]%cols>0 else 0))
        fig = plt.figure(figsize=(30,40))
        imgs = (A_i[:,:]+avg_i.reshape(-1,1)).reshape(160,120,-1)
        for i in range(1 , test_states.shape[0]+1):
            ax = fig.add_subplot(rows, cols, i)
            ax.set_title('key_pose-'+str(test_states[i-1]))
            plt.imshow(imgs[:,:,i-1],cmap='gray')
        plt.show()
    return test_states

def fetch_p_vec(key_poses,K,A_i,avg_i):
    PEI = []
    PK = []
    for i in range(K):
        indicies = np.where(key_poses==i)[0]
        PEI.append((A_i[:,indicies]+avg_i.reshape(-1,1)).reshape(160,120,-1).mean(axis=-1))
        PK.append(len(indicies)/len(key_poses))
    return np.asarray(PEI), np.asarray(PK)

def fetch_data(subject=0,angle=90,noFrame=False):
    subject = str(subject)
    while len(subject)<3:
        subject = '0'+subject
    angle = str(angle)
    while len(angle)<3:
        angle = '0' + angle
    test_dataset = None if noFrame else []
    test_preprocessed = []
    for folder in sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh/'+subject)):
        frames = None if noFrame else []
        pro_frames = []
        for file in sorted(
                os.listdir(
                    '/'.join(
                        [os.getcwd(),
                        'GaitDatasetB-silh/',
                        subject,
                        folder,
                        angle]
                    )
                )):
            if not noFrame:
                frames.append(
                cv2.imread(
                    '/'.join(
                        [os.getcwd(),
                        'GaitDatasetB-silh/',
                        subject,
                        folder,
                        angle,
                        file])
                    )
                )
            pro_frames.append(
                preprocess(
                    cv2.imread(
                        '/'.join(
                            [os.getcwd(),
                            'GaitDatasetB-silh/',
                            subject,
                            folder,
                            angle,
                            file]
                            )
                        )
                    )
                )
        if not noFrame:test_dataset.append(np.array(frames))
        test_preprocessed.append(np.moveaxis(np.array(pro_frames),0,-1))
    return test_dataset, test_preprocessed

def supervision(kmeans, angle_subject=None, override=False):
    """
    kmeans is a list of kmeans for each angle as that could be the biggest variance
    """
    if angle_subject is None:
        frames = []
        frames_name = []
        outliers = []
        for folder in tqdm(sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh'))):
            for subfolder in sorted(os.listdir( '/'.join([os.getcwd(),'GaitDatasetB-silh',folder]))):
                for angle in sorted(os.listdir( '/'.join([os.getcwd(),'GaitDatasetB-silh', folder,subfolder]))):
                    if override or not os.path.isfile('/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,'labels.pkl'])):
                        kmean = kmeans[angle]
                        for file in sorted(os.listdir( '/'.join([os.getcwd(),'GaitDatasetB-silh',folder, subfolder,angle]))):
                            if file[-3:]!="pkl":
                                frames_name.append(file)
                                frames.append(cv2.imread( '/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,file])))
                        if len(frames) < 10:
                            outliers.append('/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle]))
                        else:
                            try:
                                W_t, _, _, _, _ = get_feature_vectors(frames)
                                key_poses = graph_sort(kmean,W_t)
                                info = {}
                                for key_pose, frame_name in zip(key_poses, frames_name):
                                    info[frame_name] = key_pose
                                with open('/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,'labels.pkl']),'wb') as handle:
                                    pickle.dump(info,handle,protocol=pickle.HIGHEST_PROTOCOL)
                            except Exception as e:
                                print('/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle]))
                                raise Exception(e)
                        frames_name = []
                        frames = []
        print(outliers)
        return True
    else:
        return False
                                    
                                      
def kmean_train(subject, choice, override = False):
    if os.path.exists( '/'.join([os.getcwd(),'GaitDatasetB-silh',subject,choice])):
        if override or not os.path.isfile('Kmeans_weights.npz'):
            kmeans = {}
            for angle in sorted(os.listdir( '/'.join([os.getcwd(),'GaitDatasetB-silh',subject,choice]))):
                km = KMeans()
                imgs = []
                for file in sorted(os.listdir('/'.join([os.getcwd(), 'GaitDatasetB-silh', subject,choice,angle]))):
                    if file[-3:]!="pkl":
                        imgs.append(cv2.imread('/'.join([os.getcwd(),'GaitDatasetB-silh',subject, choice,angle,file]))) 
                W_t, _, _, _, _ = get_feature_vectors(imgs)
                km.fit(W_t)
                kmeans[angle] = [km.states, km.P]
            np.savez_compressed('Kmeans_weights',**kmeans)
            del kmeans
        kmeans_weights = np.load('Kmeans_weights.npz',allow_pickle=True)
        kmeans = {}
        for angle in sorted(os.listdir( '/'.join([os.getcwd(),'GaitDatasetB-silh',subject,choice]))):
            km = KMeans()
            km.states, km.P = kmeans_weights[angle]
            kmeans[angle] = km
        return kmeans 
    else:
        raise Exception("Could not find the specified subject and choice")
        
def fetch_labels(label_angle=None,filename="labels_cache",save=True,override=False):
    if override or not os.path.isfile(os.getcwd()+'/'+filename+'.npz'):
        labels = {}
        for folder in tqdm(sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh'))):
            for subfolder in sorted(os.listdir('/'.join([os.getcwd(), 'GaitDatasetB-silh', folder]))):
                if label_angle is None:
                    angle_set = sorted(os.listdir('/'.join([os.getcwd(), 'GaitDatasetB-silh',folder,subfolder])))
                else:
                    angle_set = [label_angle]
                for angle in angle_set:
                    label_file = '/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,'labels.pkl'])
                    if os.path.isfile(label_file):
                        labels_temp = pickle.load(open(label_file, 'rb'))
                        for file in sorted(os.listdir('/'.join([os.getcwd(),'GaitDatasetB-silh', folder, subfolder, angle]))):
                            if file[-3:]!="pkl" and file[-3:]!="npz" and file[-3:]!="npy":
                                labels['/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,file])] = labels_temp[file]
        if save:
            if label_angle is not None:
                filename += '-'+label_angle
            np.savez_compressed(filename,**labels)
    else:
        labels = np.load(filename+".npz",allow_pickle=True)
    
    return labels

def encode_data(encoder,label_angle=None,save=True,override=False):
    info = {}
    for folder in tqdm(sorted(os.listdir(os.getcwd()+'/GaitDatasetB-silh'))):
            for subfolder in sorted(os.listdir('/'.join([os.getcwd(), 'GaitDatasetB-silh', folder]))):
                if label_angle is None:
                    angle_set = sorted(os.listdir('/'.join([os.getcwd(), 'GaitDatasetB-silh',folder,subfolder])))
                else:
                    angle_set = [label_angle]
                for angle in angle_set:
                    label_file = '/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,'labels.pkl'])
                    enc_file = '/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,'enc_vec.npy'])
                    if (override or not os.path.isfile(enc_file)) and os.path.isfile(label_file):
                        labels = pickle.load(open(label_file, 'rb'))
                        labels_temp = list(enumerate(labels))
                        images = []
                        enc_vec = []
                        for file in sorted(os.listdir('/'.join([os.getcwd(),'GaitDatasetB-silh', folder, subfolder, angle]))):
                            if file[-3:]!="pkl" and file[-3:]!="npy":
                                file_name = '/'.join([os.getcwd(), 'GaitDatasetB-silh', folder, subfolder, angle, file])
                                images.append(cv2.copyMakeBorder(preprocess(cv2.imread(file_name)), 0, 0, 20, 20, cv2.BORDER_CONSTANT, (0,0,0)).reshape(160,160,1)/255.)
                        for j in range(int(np.ceil(len(labels_temp)/50.0))):
                            
                            imgs = np.zeros((50,160,160,1))
                            z = np.zeros((50),dtype=int)
                            for i, (global_i, file) in enumerate(labels_temp[50*j : min(50*(j+1), len(labels_temp)-1)]):
                                #print(i,global_i)
                                imgs[i,] = images[global_i]
                                z[i] = labels[file]
                            z_vec = keras.utils.to_categorical(z,num_classes=16)
                            enc_vec.append(encoder.predict([imgs, z_vec ], batch_size=50))
                        enc_vec = np.array(enc_vec).reshape(-1,16)[:len(labels_temp),:]
                        info['/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle])] = enc_vec
                        if save:
                            np.save('/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle,'enc_vec']),enc_vec)
                    elif os.path.isfile(enc_file):
                        info['/'.join([os.getcwd(),'GaitDatasetB-silh',folder,subfolder,angle])] = np.load(enc_file)
    return info

def encoded2timeseries(encoded_data, timestep=3, y_out = True):
    x, y = [], []
    for key, arr in encoded_data.items():
        for i in range(arr.shape[0] - timestep):
            x.append(arr[i:i+timestep,:])
            if y_out:
                y.append(arr[i+timestep,:])
    return np.array(x), np.array(y)