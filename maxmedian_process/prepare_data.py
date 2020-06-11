import os
import sys
import random
import math
from multiprocessing import Pool
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import time
from nd2reader import ND2Reader
from collections import Counter



import tifffile as tiff
import cv2
import pickle
import glob
import json
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from numba import cuda

from read_roi import read_roi_zip, read_roi_file
import matplotlib.pyplot as plt
import tifffile as tiff
import time
import gc
import numpy as np
import cv2
from datetime import datetime
import itertools


import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

import keras
from keras.models import *
from keras.models import load_model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.losses import binary_crossentropy
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def weighted_dice_coeff(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    return score


def weighted_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_true, y_pred, weight)
    return loss


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    if K.int_shape(y_pred)[1] == 128:
        kernel_size = 11
    elif K.int_shape(y_pred)[1] == 256:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 512:
        kernel_size = 21
    elif K.int_shape(y_pred)[1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size')
    averaged_mask = K.pool2d(
        y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + (1 - weighted_dice_coeff(y_true, y_pred, weight))
    return loss


def smoothen(x):
    return cv2.GaussianBlur(x, (5,5), 0)

def smoothen_parallel(x):
    return smoothen(data[x])


def adjust_baseline_parallel(x):
    return data[x] - mean_at_frame[x]

def parallel_extract_data(x):
    return np.array(data[x-1:x+2])


def parallel_resize_norm(x):
    img = cv2.resize(data[x], resize_val)
    img = img - img.min()
    img = img / img.max()
    return img

def parallel_resize(x, data, resize_val):
    return cv2.resize(data[x], resize_val)


def parallel_diff_segments(x):
    return (np.max(data[x:x+wlen], axis=0) - np.median(data[x:x+wlen], axis=0))

def normalize(X):
    
    
    X = X - X.min()
    X = X / X.max()
    X = X*255
    X = X.astype('uint8')
    
    return X

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_prob(vid, mag, weight_path):
    
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    tf.Session(config = session_config)

           
    input_size = (128,128,3)

    inputs = Input(input_size)
    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16hippocampus_20181126_602086

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    new_model = multi_gpu_model(model, gpus=2)
    crop = 1
    if(crop == 1):    
        vid = vid[:,5:-5,5:-5]

    if(mag == 40):
        wlen = 20
        GW = 17
    elif(mag == 20):
        wlen = 40
        GW = 15
    else:
        wlen = 50
        GW = 13
    

    H = vid.shape[1]
    W = vid.shape[2]

    pool_out = []
    for i in range(vid.shape[0]):
        pool_out.append(cv2.GaussianBlur(vid[i], (GW,GW), 0))
    smoothened = np.array(pool_out, 'float32')

    del pool_out
    gc.collect()

    mean_at_frame = np.mean(np.mean(smoothened,axis=1),axis=1)

    pool_out = []
    for i in range(vid.shape[0]):
        pool_out.append(smoothened[i] - mean_at_frame[i])
    adj = np.array(pool_out, 'float32')

    del smoothened
    gc.collect()

    pool_out = []
    for i in range(0, vid.shape[0], wlen):
        pool_out.append(np.max(adj[i:i+wlen], axis=0) - np.median(adj[i:i+wlen], axis=0))
    vImg = np.array(pool_out, 'float32')

    del pool_out, adj
    gc.collect()

    pool_out = []
    for i in range(1, vImg.shape[0]-1):
        pool_out.append(np.array(vImg[i-1:i+2]))
    vImg3 = np.array(pool_out, 'float32')

    del pool_out
    gc.collect()



    vImg3 = vImg3.transpose((0,2,3,1))

    pool_out = []
    for i in range(vImg3.shape[0]):
        img = cv2.resize(vImg3[i], (128, 128))
        img = img - img.min()
        img = img / img.max()
        img = img * 255
        img = img.astype('uint8')
        pool_out.append(img)
    xdata = np.array(pool_out, 'uint8')    


    xtest = xdata

    # Loss = '../../CA4_TransferLearning/BestLoss_CA4.hdf5'
    Loss = weight_path

    new_model.load_weights(Loss)

    ypred = new_model.predict(xtest)

    data=ypred
    resize_val = (W, H)
    pool_out = []
    for i in range(data.shape[0]):
        pool_out.append(parallel_resize(i, data, resize_val))
    ypred = np.array(pool_out, 'float32')

    if(crop == 1):
        blk = np.zeros((vImg.shape[0]-2, vImg.shape[1]+10, vImg.shape[2] + 10), dtype='float32')
        blk[:,5:-5, 5:-5] = ypred
    else:
        blk = ypred
#     outname = outpath + '/' + path_leaf(inputname)
#     tiff.imsave(outname, blk)
    
    return blk
