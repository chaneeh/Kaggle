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

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

def get_optimize_op(global_step, learning_rate, config, loss_opt):
        """
        Need to override if you want to use different optimization policy.
        :param learning_rate:
        :param global_step:
        :return: (learning_rate, optimizer) tuple
        """
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   decay_steps=config.opt_decay_steps,
                                                   decay_rate=config.opt_decay_rate,
                                                   staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
            optimize_op = optimizer.minimize(loss_opt, global_step, colocate_gradients_with_ops=True)
        return learning_rate, optimize_op




def get_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())


def U_net(X, config, re=False):
    with tf.variable_scope('u_', reuse=re) as scope:
        ### Unit 1 ###
        with tf.name_scope('Unit1'):
            W1_1 = get_variable("W1_1", [3, 3, 3, 16])
            Z1 = tf.nn.conv2d(X, W1_1, strides=[1, 1, 1, 1], padding='SAME')
            A1 = tf.nn.relu(Z1)
            W1_2 = get_variable("W1_2", [3, 3, 16, 16])
            Z2 = tf.nn.conv2d(A1, W1_2, strides=[1, 1, 1, 1], padding='SAME')
            A2 = tf.nn.relu(Z2)
            P1 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ### Unit 2 ###
        with tf.name_scope('Unit2'):
            W2_1 = get_variable("W2_1", [3, 3, 16, 32])
            Z3 = tf.nn.conv2d(P1, W2_1, strides=[1, 1, 1, 1], padding='SAME')
            A3 = tf.nn.relu(Z3)
            W2_2 = get_variable("W2_2", [3, 3, 32, 32])
            Z4 = tf.nn.conv2d(A3, W2_2, strides=[1, 1, 1, 1], padding='SAME')
            A4 = tf.nn.relu(Z4)
            P2 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ### Unit 3 ###
        with tf.name_scope('Unit3'):
            W3_1 = get_variable("W3_1", [3, 3, 32, 64])
            Z5 = tf.nn.conv2d(P2, W3_1, strides=[1, 1, 1, 1], padding='SAME')
            A5 = tf.nn.relu(Z5)
            W3_2 = get_variable("W3_2", [3, 3, 64, 64])
            Z6 = tf.nn.conv2d(A5, W3_2, strides=[1, 1, 1, 1], padding='SAME')
            A6 = tf.nn.relu(Z6)
            P3 = tf.nn.max_pool(A6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ### Unit 4 ###
        with tf.name_scope('Unit4'):
            W4_1 = get_variable("W4_1", [3, 3, 64, 128])
            Z7 = tf.nn.conv2d(P3, W4_1, strides=[1, 1, 1, 1], padding='SAME')
            A7 = tf.nn.relu(Z7)
            W4_2 = get_variable("W4_2", [3, 3, 128, 128])
            Z8 = tf.nn.conv2d(A7, W4_2, strides=[1, 1, 1, 1], padding='SAME')
            A8 = tf.nn.relu(Z8)
            P4 = tf.nn.max_pool(A8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ### Unit 5 ###
        with tf.name_scope('Unit5'):
            W5_1 = get_variable("W5_1", [3, 3, 128, 256])
            Z9 = tf.nn.conv2d(P4, W5_1, strides=[1, 1, 1, 1], padding='SAME')
            A9 = tf.nn.relu(Z9)
            W5_2 = get_variable("W5_2", [3, 3, 256, 256])
            Z10 = tf.nn.conv2d(A9, W5_2, strides=[1, 1, 1, 1], padding='SAME')
            A10 = tf.nn.relu(Z10)
            ### Unit 6 ###
        with tf.name_scope('Unit6'):
            W6_0 = get_variable("W6_0", [2, 2, 128, 256])
            U1 = tf.nn.conv2d_transpose(A10, W6_0, output_shape=[config.batch_size, 32, 32, 128], strides=[1, 2, 2, 1])
            #U1 = tf.layers.conv2d_transpose(A10, filters = 128, kernel_size = 2, strides = 2, padding = 'SAME')
            #U1 = tf.concat(3, [U1, A8])
            U1 = tf.concat([U1, A8], axis=-1)
            
            W6_1 = get_variable("W6_1", [3, 3, 256, 128])
            Z11 = tf.nn.conv2d(U1, W6_1, strides=[1, 1, 1, 1], padding='SAME')
            A11 = tf.nn.relu(Z11)

            W6_2 = get_variable("W6_2", [3, 3, 128, 128])
            Z12 = tf.nn.conv2d(A11, W6_2, strides=[1, 1, 1, 1], padding='SAME')
            A12 = tf.nn.relu(Z12)
        ### Unit 7 ###
        with tf.name_scope('Unit7'):
            W7_0 = get_variable("W7_0", [2, 2, 64, 128])
            U2 = tf.nn.conv2d_transpose(A12, W7_0, output_shape=[config.batch_size, 64, 64, 64], strides=[1, 2, 2, 1])
            # U2 = tf.layers.conv2d_transpose(A12, filters = 64, kernel_size = 2, strides = 2, padding = 'SAME')
            #U2 = tf.concat(3, [U2, A6])
            U2 = tf.concat([U2, A6], axis=-1)

            W7_1 = get_variable("W7_1", [3, 3, 128, 64])
            Z13 = tf.nn.conv2d(U2, W7_1, strides=[1, 1, 1, 1], padding='SAME')
            A13 = tf.nn.relu(Z13)

            W7_2 = get_variable("W7_2", [3, 3, 64, 64])
            Z14 = tf.nn.conv2d(A13, W7_2, strides=[1, 1, 1, 1], padding='SAME')
            A14 = tf.nn.relu(Z14)
        ### Unit 8 ###
        with tf.name_scope('Unit8'):
            W8_0 = get_variable("W8_0", [2, 2, 32, 64])
            U3 = tf.nn.conv2d_transpose(A14, W8_0, output_shape=[config.batch_size, 128, 128, 32], strides=[1, 2, 2, 1])
            # U3 = tf.layers.conv2d_transpose(A14, filters = 32, kernel_size = 2, strides = 2, padding = 'SAME')
            #U3 = tf.concat(3, [U3, A4])
            U3 = tf.concat([U3, A4], axis=-1)
            
            W8_1 = get_variable("W8_1", [3, 3, 64, 32])
            Z15 = tf.nn.conv2d(U3, W8_1, strides=[1, 1, 1, 1], padding='SAME')
            A15 = tf.nn.relu(Z15)

            W8_2 = get_variable("W8_2", [3, 3, 32, 32])
            Z16 = tf.nn.conv2d(A15, W8_2, strides=[1, 1, 1, 1], padding='SAME')
            A16 = tf.nn.relu(Z16)
        ### Unit 9 ###
        with tf.name_scope('Unit9'):
            W9_0 = get_variable("W9_0", [2, 2, 16, 32])
            U4 = tf.nn.conv2d_transpose(A16, W9_0, output_shape=[config.batch_size, 256, 256, 16], strides=[1, 2, 2, 1])
            # U4 = tf.layers.conv2d_transpose(A16, filters = 16, kernel_size = 2, strides = 2, padding = 'SAME')
            #U4 = tf.concat(3, [U4, A2])
            U4 = tf.concat([U4, A2], axis=-1)
            
            W9_1 = get_variable("W9_1", [3, 3, 32, 16])
            Z17 = tf.nn.conv2d(U4, W9_1, strides=[1, 1, 1, 1], padding='SAME')
            A17 = tf.nn.relu(Z17)

            W9_2 = get_variable("W9_2", [3, 3, 16, 16])
            Z18 = tf.nn.conv2d(A17, W9_2, strides=[1, 1, 1, 1], padding='SAME')
            A18 = tf.nn.relu(Z18)
        ### Unit 10 ###
        with tf.name_scope('out_put'):
            W10 = get_variable("W10", [1, 1, 16, 1])
            Z19 = tf.nn.conv2d(A18, W10, strides=[1, 1, 1, 1], padding='SAME')
            A19 = tf.nn.sigmoid(Z19)

        return Z19, A19  # (logit, log)

