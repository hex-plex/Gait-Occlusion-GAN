#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from numpy import savez_compressed
import cv2
from matplotlib import pyplot as plt




# In[7]:


data = np.load('maps_256.npz')
src_images, tar_images = data['arr_0'], data['arr_1']
src_images.shape


# In[8]:


for i in range(3):
    plt.subplot(2,3,1+i)
    plt.axis('off')
    plt.imshow(src_images[i].astype(np.uint8))
for i in range(3):
    plt.subplot(2,3,i+4)
    plt.axis('off')
    plt.imshow(tar_images[i].astype(np.uint8))
plt.show()


# In[9]:


def define_discriminator(img_shape):
    init = keras.initializers.RandomNormal(stddev=0.02)
    in_src_image = keras.layers.Input(shape=img_shape)
    in_target_image = keras.layers.Input(shape=img_shape)
    merged = keras.layers.Concatenate()([in_src_image, in_target_image])
    d = keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    
    d = keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    
    d = keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    
    d = keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = keras.layers.BatchNormalization()(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)
    
    d = keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = keras.layers.Activation('sigmoid')(d)
    
    model = keras.models.Model([in_src_image, in_target_image], patch_out)
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# In[10]:


def define_encoder_block(layer_in, n_filters, batchnorm = True):
    init = keras.initializers.RandomNormal(stddev=0.02)
    g = keras.layers.Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = keras.layers.BatchNormalization()(g, training=True)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    init = keras.initializers.RandomNormal(stddev=0.02)
    g = keras.layers.Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = keras.layers.BatchNormalization()(g, training=True)
    if dropout:
        g = keras.layers.Dropout(0.5)(g, training=True)
    g = keras.layers.Concatenate()([g, skip_in])
    g = keras.layers.Activation('relu')(g)
    return g

def define_generator(img_shape=(256,256,3)):
    init = keras.initializers.RandomNormal(stddev=0.02)
    
    in_image = keras.layers.Input(shape=img_shape)
    
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    
    b = keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = keras.layers.Activation('relu')(b)
    
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    
    g = keras.layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = keras.layers.Activation('tanh')(g)
    model = keras.models.Model(in_image, out_image)
    return model


# In[11]:


def define_gan(g_model, d_model, image_shape):
    d_model.trainable = False
    in_src = keras.layers.Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = keras.models.Model(in_src, [dis_out, gen_out])
    opt = keras.optimizers.Adam(lr=0.002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    return model


# In[12]:


def load_real_samples(filename):
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# In[13]:


def summarize_performance(step, g_model, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    
    for i in range(n_samples):
        plt.subplot(3, n_samples, i+1)
        plt.axis('off')
        plt.imshow(X_realA[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, i + n_samples + 1)
        plt.axis('off')
        plt.imshow(X_fakeB[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, i + 2*n_samples + 1)
        plt.axis('off')
        plt.imshow(X_realB[i])
        
    filename1 = 'plot_%06d.png' % (step+1)
    plt.savefig(filename1)
    plt.close()
    
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# In[14]:


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    
    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _ , _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
        if (i+1)%(bat_per_epo*10) == 0:
            summarize_performance(i,g_model,dataset)


# In[15]:


dataset = load_real_samples('maps_256.npz')
print('loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]


# In[16]:


d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

gan_model = define_gan(g_model, d_model, image_shape)


# In[17]:


train(d_model, g_model, gan_model, dataset)


# In[ ]:




