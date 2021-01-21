import tensorflow as tf
import keras
from keras import backend as K
from keras.utils.vis_utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import pickle
from tqdm import tqdm
import gait

image_vec_shape = 12
gait_key_poses = 16
conditional_vec_shape = 10
pose_image_shape = (160,160,1)
batch_size = 50
full_vec_shape = image_vec_shape+conditional_vec_shape
n_timesteps = 3
angle = '090'

def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, image_vec_shape), mean=0., stddev=1.0)
    return mu + K.exp(log_var/2.)*eps
def encoder_model():
    x_in = keras.layers.Input(name="x_image",
                              shape=pose_image_shape)
    condition_in = keras.layers.Input(name="condition_hotshot",
                                      shape=(gait_key_poses,))
    condi_vec1 = keras.layers.Dense(8,activation='relu',name="condition_dense_1")(condition_in)
    condi_vec = keras.layers.Dense(4,activation='relu',name="condition_dense_2")(condi_vec1)
    x = keras.layers.Conv2D(filters=16,
                            kernel_size=(5,5),
                            strides=(3,3),
                            activation='relu',
                            name="x_conv_1")(x_in)
    x = keras.layers.BatchNormalization(name="x_batch_norm_1")(x,training=True)
    x = keras.layers.Conv2D(filters=32,
                            kernel_size=(5,5),
                            strides=(3,3),
                            activation='relu',
                            name="x_conv_2")(x)
    x = keras.layers.BatchNormalization(name="x_batch_norm_2")(x, training=True)
    x = keras.layers.Conv2D(filters=64,
                            kernel_size=(5,5),
                            strides=(3,3),
                            activation='relu',
                            name='x_conv_3')(x)
    x = keras.layers.BatchNormalization(name="x_batch_norm_3")(x, training=True)
    x = keras.layers.Flatten(name="x_flatten")(x)
    x = keras.layers.Dense(336, activation='relu',name="x_dense_final")(x)
    x_vec = keras.layers.concatenate([x,condi_vec], name="concat_xvec")
    x_vec = keras.layers.Dense(32, activation='relu',name="xvec_dense_final")(x_vec)
    mu = keras.layers.Dense(image_vec_shape, activation='linear',name="mu_dense")(x_vec)
    log_var = keras.layers.Dense(image_vec_shape, activation='linear', name="logvar_dense")(x_vec)
    z = keras.layers.Lambda(sampling, output_shape=(image_vec_shape,), name="lambda_sampling")([mu,log_var])
    z_cond = keras.layers.concatenate([z, condi_vec],name="concat_zcond")
    
    encoder = keras.models.Model([x_in,condition_in], z_cond)
    
    z1 = keras.layers.Dense(32, activation='relu', name="z_dense_1")(z_cond)
    z1 = keras.layers.Dense(336, activation='relu', name="z_dense_2")(z1)
    z1 = keras.layers.Dense(1024, activation='relu', name="z_dense_3")(z1)
    zim = keras.layers.Reshape((4,4,64),name="reshape_image")(z1)
    zim = keras.layers.Conv2DTranspose(filters=32,
                                       kernel_size=(7,7),
                                       strides=(3,3),
                                       activation='relu',
                                       name="z_convT_1")(zim)
    zim = keras.layers.Dropout(0.1, name="z_dropout_1")(zim, training=True)
    zim = keras.layers.Conv2DTranspose(filters=16,
                                       kernel_size=(7,7),
                                       strides=(3,3),
                                       activation='relu',
                                       name="z_convT_2")(zim)
    zim = keras.layers.Dropout(0.1, name="z_dropout_2")(zim, training=True)
    zim = keras.layers.Conv2DTranspose(filters=8,
                                       kernel_size=(7,7),
                                       strides=(3,3),
                                       activation='relu',
                                       name="z_convT_3")(zim)
    zim = keras.layers.Dropout(0.1, name="z_dropout_3")(zim, training=True)
    y = keras.layers.Conv2D(filters=1,
                            kernel_size=(1,1),
                            activation='tanh',
                            name="image_out")(zim)
    
    #decoder = keras.models.Model(z_cond_in, y)
    full_model = keras.models.Model([x_in, condition_in], y)
    
    reconstruction_loss = K.sum(K.sum(keras.losses.binary_crossentropy(x_in, y), axis=1),axis=1)
    kl_loss = 0.5*K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis=-1)
    cvae_loss = reconstruction_loss + kl_loss
    
    full_model.add_loss(cvae_loss)
    full_model.compile(optimizer='adam', loss=None)
    ## Ignore the missing from loss dictionary error
    return encoder, full_model

CVAE, CVAE_FULL = encoder_model()
tb = keras.callbacks.TensorBoard(log_dir="logs")
mc = keras.callbacks.ModelCheckpoint(filepath="weights/CVAE_FULL.h5")
class DataGenerator(keras.utils.Sequence):
    def __init__(self, files, labels, preprocess=None, batch_size = 50, dim=(160,160), n_channels=1, n_classes=16, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.files = files
        self.preprocess = preprocess or (lambda x : x)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.files)/self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        files_temp = [self.files[k] for k in indexes]
        
        X,z = self.__data_generation(files_temp)
        return [X,z], None
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        z = np.empty((self.batch_size),dtype=int)
        for i,file in enumerate(files_temp):
            X[i,] = cv2.copyMakeBorder(self.preprocess(cv2.imread(file)), 0, 0, 20, 20, cv2.BORDER_CONSTANT, (0,0,0)).reshape(*self.dim, self.n_channels)/255.
            z[i] = self.labels[file]
        return X, keras.utils.to_categorical(z, num_classes=self.n_classes)
    
labels = gait.fetch_labels(label_angle=angle,save=False,override=True)
files = [filename for filename in labels]

test_files = []
test_partition = np.random.randint(0,len(files),size=3000)
for i in test_partition:
    test_files.append(files[i])
for i in sorted(test_partition)[::-1]:
    del files[i]
    
train_data = DataGenerator(files,labels,preprocess=gait.preprocess)
valid_data = DataGenerator(test_files, labels, preprocess=gait.preprocess)
if os.path.isfile(os.getcwd()+"weights/CVAE_FULL.h5"):
	CVAE_FULL.load_weights("weights/CVAE_FULL.h5")
	print("Successfully loaded the model")

history = CVAE_FULL.fit_generator(generator=train_data,
                                  validation_data = valid_data,
                                  steps_per_epoch = len(files)//batch_size,
                                  epochs=1000,
                                  validation_steps = len(test_files)//batch_size,
				  callbacks=[tb,mc],
                                  use_multiprocessing=True,
                                  workers=4)
CVAE_FULL.save_weights("weights/CVAE_FULL.h5")
