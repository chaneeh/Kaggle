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

def get_net_input_size(image_size, num_block):
    network_input_size = image_size
    for _ in range(num_block):
        #assert network_input_size % 2 == 0, network_input_size
        network_input_size = (network_input_size + 4) // 2
    network_input_size += 4
    for _ in range(num_block):
        network_input_size = network_input_size * 2 + 4
    return network_input_size


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def to_csv(pred_labels, origin_size, test_ids, csv_name, cut_off):
    new_test_ids = []
    rles = []
    preds_test_upsampled = []
    for i in range(len(pred_labels)):
        preds_test_upsampled.append(resize(np.squeeze(pred_labels[i]),
                                           (origin_size[i][0], origin_size[i][1]),
                                           mode='constant', preserve_range=True))
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n], cutoff=cut_off))
        if (len(rle) == 0):
            rle = [[0,0]]
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(csv_name, index=False)



def loss_function(y_pred, y_true):
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return cost


def mean_iou(y_pred, y_true):
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score