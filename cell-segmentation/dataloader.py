# original code from https://github.com/jfpuget/DSB_2018.git
# reference source: 
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/54426#latest-482147
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/54742#latest-322587


import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.measure import label
from skimage import transform
from scipy import ndimage

from utils import *


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

def random_crop(data, mask, w, h, padding=0):
    img_w, img_h = data.shape[:2]
    
    x = random.randint(0, img_w - w)
    y = random.randint(0, img_h - h)
    
    return crop(data, mask, x, y, w, h, padding=padding)
    
def crop(data, mask, x, y, w, h, padding=0):
    img_w, img_h = data.shape[:2]
    
    return crop_mirror(data, mask, x, y, w, h, padding)

def crop_mirror(img, mask, x, y, w, h, padding=0):
    mirror_padded = mirror_pad(img, padding)
    cropped_img = mirror_padded[x:x+w+padding*2, y:y+h+padding*2]
    cropped_mask = mask[x:x+w, y:y+h]
    return cropped_img, cropped_mask

def mirror_pad(img, padding):
    if len(img.shape) == 3: #but all train datas are 3 dim!
        padded_img = np.array([np.pad(ch, padding, 'reflect') for ch in img.transpose((2, 0, 1))]).transpose((1, 2, 0))
    else:
        channel = img.shape[-1]
        padded_img = np.pad(img, padding, 'reflect') [:,:,padding:padding+channel]
    
    return padded_img

def rotate_flip(img, rotate_angle, flip_boolean):
    rows, cols = img.shape[:2]
    M= cv2.getRotationMatrix2D((rows/2, cols/2), 90*rotate_angle, 1.)
    dst = cv2.warpAffine(img, M, (rows, cols))
                
    if (flip_boolean):
        dst = cv2.flip(dst, 0)
    return dst

class Dataloader(object):
    def __init__(self, config, dup=4):
        self.config = config
        #train_ids = next(os.walk(config.train_dir))[1][:8]
        train_ids = next(os.walk(config.train_dir))[1][:9]
        #test_ids = next(os.walk(config.test_dir))[1][:100]

        self.num_block = config.unet_step_size
        self.inp_size = get_net_input_size(config.input_height, self.num_block)
        self.pad_size = (self.inp_size - config.input_height) // 2

        images_shape = (len(train_ids)*dup, config.input_width + 2*self.pad_size, config.input_height + 2*self.pad_size, config.img_channel)
        labels_shape = (len(train_ids)*dup, config.input_width, config.input_height, config.label_channel)
        
        images = np.zeros(images_shape, dtype=np.float32)
        labels = np.zeros(labels_shape, dtype=np.float32)
    
        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()

        for n, id_ in enumerate(train_ids):
            n = n*dup
            path = config.train_dir + id_
            img = imread(path + '/images/' + id_ + '.png')[:, :, :config.img_channel]
            #img = imread(path + '/images/' + id_ + '.png')[:, :, :config.img_channel]
            #img = img / np.max(img)
              
            img = resize(img, (config.input_height, config.input_width), mode='constant',
                                              preserve_range=True) / 255.0
        
            mask = np.zeros((img.shape[0], img.shape[1], config.label_channel), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (config.input_height, config.input_width), mode='constant',       preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
                
            mask = mask / 255.0
            
         
            
            crop_img_, mask_img_ = random_crop(np.copy(img), np.copy(mask), config.input_width, config.input_height, padding=self.pad_size)
    
    
            for i in range(dup):
                copy_img = np.copy(crop_img_)
                copy_lab = np.copy(mask_img_)
                                
                rotate_angle = i
                flip_boolean = 0
                
                images[n+i] = rotate_flip(copy_img, rotate_angle, flip_boolean)
                labels[n+i] = np.expand_dims(rotate_flip(copy_lab, rotate_angle, flip_boolean), axis=-1)
                
               
         

                  
       
        images_shape = images.shape
        labels_shape = labels.shape

        self.train_ids = train_ids
        

        
        self.image_shape = images_shape
        self.batch = config.batch_size
        self.len_images = self.image_shape[0]

        self.X_train = images
        self.Y_train = labels
        
        print(images_shape)
        print('Done!')

    def prepare_test_images(self):
        del self.images
        del self.labels
        
        test_ids = next(os.walk(self.config.test_dir))[1]
        test_shape   = (len(test_ids), self.config.input_width + self.pad_size*2, self.config.input_height + self.pad_size*2, self.config.img_channel)
        #test_shape   = (len(test_ids), self.config.input_width, self.config.input_height, self.config.img_channel)
        
    
        X_test = np.zeros(test_shape, dtype=np.float32)
        sizes_test = []

        print('Getting and resizing test images ... ')
        sys.stdout.flush()

        for n, id_ in enumerate(test_ids):
            path = self.config.test_dir + id_
            if (len(imread(path + '/images/' + id_ + '.png').shape) == 2):
                f_copy = np.copy(np.expand_dims(imread(path + '/images/' + id_ + '.png'), -1))
                img = np.concatenate([f_copy, np.copy(f_copy), np.copy(f_copy)], axis = -1)
            else:
                img = imread(path + '/images/' + id_ + '.png')[:, :, :self.config.img_channel]
            sizes_test.append([img.shape[0], img.shape[1]])
            #print(img.shape)
            if (img.shape[0] < 256):
                print("width smaller than 256!")
            if (img.shape[1] < 256):
                print("height smaller than 256!")
            img = resize(img, (self.config.input_height, self.config.input_width), mode='constant', preserve_range=True)
            img = img / 255.0
            X_test[n] = mirror_pad(img, self.pad_size)
            #X_test[n] = img
        self.test_ids = test_ids
        self.test_sizes = sizes_test
        self.X_test = X_test
        
        
        

    def shuffle(self):
        p = np.random.permutation(len(self.X_train))
        self.images = self.X_train[p]
        self.labels = self.Y_train[p]
        del self.X_train
        del self.Y_train
        del p
       

    def next_batch(self, iters):
        count_from = (self.batch * iters) % self.len_images
        count_to = (self.batch * iters + self.batch) % self.len_images

        if (count_from >= count_to):
            return self.images[self.len_images - self.batch: self.len_images], self.labels[self.len_images - self.batch: self.len_images]
            
        else:
            return self.images[count_from:count_to], self.labels[count_from:count_to]

    def data_aug(self, image, label, config, angel=30, resize_rate=0.9):
        flip = random.randint(0, 1)
        size = image.shape[0]
        rsize = random.randint(np.floor(resize_rate * size), size)
        w_s = random.randint(0, size - rsize)
        h_s = random.randint(0, size - rsize)
        sh = random.random() / 2 - 0.25
        rotate_angel = random.random() / 180 * np.pi * angel
        
        afine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angel)

        image = transform.warp(image, inverse_map=afine_tf, mode='constant')
        label = transform.warp(label, inverse_map=afine_tf, mode='constant')

        #image = image[w_s:w_s + size, h_s:h_s + size, :]
        #label = label[w_s:w_s + size, h_s:h_s + size]

        if flip:
            image = image[:, ::-1, :]
            label = label[:, ::-1]
        
        return image, label





    def test_images(self):
        return self.X_test, self.test_sizes, self.test_ids
        #(65, 256, 256, 3)

    