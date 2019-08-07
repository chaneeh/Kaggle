from memory_profiler import profile
import os
import sys
import numpy as np
import random
import math
import warnings
import cv2
#import PyQt4
import matplotlib
matplotlib.use('Agg')
#matplotlib.get_backend()
#matplotlib.use('qt4Agg')
import matplotlib.pyplot as plt
#matplotlib.pyplot.ion()

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.measure import label
from skimage import transform


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

train_dir = "stage1_dataset/"
test_dir = "stage2_dataset/stage2_files/"

#train_ids = next(os.walk(train_dir))[1]
#test_ids = next(os.walk(test_dir))[1]



@profile
def my_func():
    c = []
    #a = [1] * (10 ** 6)
    #b = [2] * (2 * 10 ** 7)
    b_ = np.array([[2]] * (2 * 10 ** 7))
    bb = b_
    
    p = np.random.permutation(len(b_))
    print(type(p[0]))
    d = b_[p]
    #c.append(b)
    del b_
    del p
    print(len(d))
    
    '''
    print(train_ids[:10])
    a = []
    
    path = train_dir + train_ids[0]
    img = cv2.imread(path + '/images/' + train_ids[0] + '.png', cv2.IMREAD_COLOR)
    _img = np.copy(img)
    a.append(img)
    
    
    path = train_dir + train_ids[2]
    img3 = cv2.imread(path + '/images/' + train_ids[2] + '.png', cv2.IMREAD_COLOR)
    _img3 = np.copy(img3)
    _img3[0][0][0] = 100000
    
    
    images = []
    for idx, id_ in enumerate(train_ids[:300]):
        path = train_dir + id_
        img10 = cv2.imread(path + '/images/' + id_ + '.png', cv2.IMREAD_COLOR)
        img11 = np.copy(img10)
        img11[0][0] = 100
        images.append(img10)
        
    del images
    '''
        
       
    
    
    
    

if __name__ == '__main__':
    my_func()
