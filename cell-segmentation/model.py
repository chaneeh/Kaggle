# original code from https://github.com/jfpuget/DSB_2018.git
# reference source: 
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/54426#latest-482147
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/54742#latest-322587


from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np
from six.moves import xrange

#from ops import *
from ops import *
from utils import *
from dataloader import Dataloader


def double_conv(net, nb_filter, scope):
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_1' % scope)
        net = slim.dropout(net)
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_2' % scope)
        return net


class UCell(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.epoch = config.epoch

        self.batch_size = config.batch_size

        self.input_height = config.input_height
        self.input_width = config.input_width
        self.img_channel = config.img_channel
        self.label_channel = config.label_channel

        
        self.num_block = config.unet_step_size
        self.inp_size = get_net_input_size(self.input_height, self.num_block)
        self.pad_size = (self.inp_size - self.input_height) // 2
        
        self.input_shape = [config.batch_size, self.input_width + self.pad_size*2, self.input_height + self.pad_size*2, 3]
        #self.input_shape = [config.batch_size, self.input_width, self.input_height, 3]
        self.mask_shape  = [config.batch_size, self.input_width, self.input_height, 1]
        
        self.is_save     = config.is_save
        self.is_training = config.is_training
        self.unet_weight = config.unet_weight
        
        
     

        self.model_name = config.model_name
        self.ckpt_model_name = config.ckpt_model_name
        self.dataset_name = config.dataset_name
        self.train_dir = config.train_dir
        self.test_dir = config.test_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.csv_name = config.csv_name

        self.dataloader = Dataloader(config)
        self.dataloader.shuffle()

        self.build_model()

    def build_model(self):
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.inputs = tf.placeholder(tf.float32, shape = self.input_shape, name='image_input')
        self.outputs = tf.placeholder(tf.float32, shape = self.mask_shape, name='mask_output')

        self.labels_logits, self.labels_log = self.U_net(self.inputs, self.config)
        
        w = 1.0
        if self.unet_weight:
            w = self.weight

        self.u_loss_p = sigmoid_cross_entropy_with_logits(self.labels_logits, self.outputs)
        self.u_lossp = tf.reduce_mean(self.u_loss_p)

            
        self.u_loss_ = tf.nn.weighted_cross_entropy_with_logits(
            targets=self.outputs,
            logits=self.labels_logits,
            pos_weight=w
        )
        self.u_loss = tf.reduce_mean(self.u_loss_)
        
        #self.u_loss = self.u_lossp
        self.u_loss_prev = self.u_lossp

        ###################################################################
        ########################  l_r update  #############################
        ###################################################################
        self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False, name='l_lr')
        self.lr_update = tf.assign(self.learning_rate, tf.clip_by_value(self.learning_rate*self.config.lr_decay, 0, 1))
        self.lr_update_init = tf.assign(self.learning_rate, self.config.learning_rate)
        
        
        
        trainable_vars = tf.trainable_variables()
        self.u_vars = [var for var in trainable_vars if 'u_' in var.name]
        print("length of trainable_vars is %d ", len(self.u_vars))

        self.saver = tf.train.Saver()
        
    def U_net(self, X, config, re=False):
        with tf.variable_scope('u_', reuse=re) as scope:
            weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=config.net_init_stddev)
            batch_norm_params = {
                'is_training': self.is_training,
                'center': True,
                'scale': True,
                'decay': config.net_bn_decay,
                'epsilon': config.net_bn_epsilon,
                'fused': True
                 #'zero_debias_moving_mean': True
                }

            dropout_params = {
                'keep_prob': config.net_dropout_keep,
                'is_training': self.is_training,
                }

            conv_args = {
                'padding': 'VALID',
                'weights_initializer': weight_init,
                'normalizer_fn': slim.batch_norm,
                'normalizer_params': batch_norm_params,
                'activation_fn': tf.nn.elu,
                'weights_regularizer': slim.l2_regularizer(0.0001)
                }

            net = X

            features = []
            with slim.arg_scope([slim.convolution, slim.conv2d_transpose], **conv_args):
                with slim.arg_scope([slim.dropout], **dropout_params):
                    base_feature_size = config.unet_base_feature
                    max_feature_size = base_feature_size * (2 ** self.num_block)

                    # down sampling steps
                    for i in range(self.num_block):
                        net = double_conv(net, int(base_feature_size*(2**i)), scope='down_conv_%d' % (i + 1))
                        features.append(net)
                        net = slim.max_pool2d(net, [2, 2], 2, padding='VALID', scope='pool%d' % (i + 1))

                    # middle
                    net = double_conv(net, max_feature_size, scope='middle_conv_1')

                    # upsampling steps
                    for i in range(self.num_block):
                        # up-conv
                        net = slim.conv2d_transpose(net,int(max_feature_size/(2**(i+1))),[2,2], 2,scope='up_trans_conv_%d' % (i + 1))

                        # get lower layer's feature
                        ############################################################
                        # change tensor_var.shape => tensor_var.get_shape().as_list()
                        ##############################################################
                        down_feat = features.pop()
                        assert net.get_shape().as_list()[3] == down_feat.get_shape().as_list()[3], '%d, %d, %d' % (i, net.get_shape().as_list()[3], down_feat.get_shape().as_list()[3])

                        y, x = [int(down_feat.get_shape().as_list()[idx] - net.get_shape().as_list()[idx]) // 2 for idx in [1, 2]]
                        h, w = map(int, net.get_shape().as_list()[1:3])
                        down_feat = tf.slice(down_feat, [0, y, x, 0], [-1, h, w, -1])

                        net = tf.concat([down_feat, net], axis=-1)
                        #net = tf.concat(3, [down_feat, net])
                        print(net.get_shape().as_list())
                        net = double_conv(net, int(max_feature_size/(2**(i+1))), scope='up_conv_%d' % (i + 1))

            # original paper : one 1x1 conv
            net = slim.convolution(net, 1, [1, 1], 1, scope='final_conv',
                                   activation_fn=None,
                                   padding='SAME',
                                   weights_initializer=weight_init)

            self.logit_ = net
            self.output_ = tf.nn.sigmoid(net, 'visualization')
        
          
        return self.logit_, self.output_

    def train(self, config):
        #global_step = tf.Variable(0, trainable=False)
        #learning_rate_v, u_optim = get_optimize_op(global_step=global_step,
        #                                           learning_rate=config.learning_rate,
        #                                           config=self.config,
        #                                           loss_opt=self.u_loss_prev)
        
        
        u_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1) \
            .minimize(self.u_loss, var_list=self.u_vars)
            
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            self.sess.run(tf.initialize_all_variables())

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
        lr_update_init = self.sess.run([self.lr_update_init])
        
        for epoch in xrange(config.epoch):
            print("epoch: %d  learning_rate: %.8f" %(epoch, self.sess.run(self.learning_rate)))
            batch_idxs = int(self.dataloader.len_images / self.batch_size)
            #batch_idxs = 10
            if self.dataloader.len_images % self.batch_size != 0:
                batch_idxs = batch_idxs + 1
            for idx in xrange(0, batch_idxs):
                #batch 8 -> 84, batch 16 -> 42
                batch_images, batch_labels = self.dataloader.next_batch(idx)
                
                _, u_error, u_loss_prev = self.sess.run([u_optim, self.u_loss, self.u_loss_prev], feed_dict={
                                                       self.inputs: batch_images,
                                                       self.outputs: batch_labels})
                counter += 1
                if np.mod(counter, 8) == 0:
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, u_loss: %.8f, u_loss_prev: %.8f" \
                        % (epoch, config.epoch, idx, batch_idxs, time.time() - start_time, u_error, u_loss_prev))

                
                
            lr_update = self.sess.run([self.lr_update])
            #step, lr = self.sess.run([global_step, learning_rate_v])
            
            if (epoch == 0) and (self.is_save):
                self.save(config.checkpoint_dir, counter)
            if (epoch == 1) and (self.is_save):
                self.save(config.checkpoint_dir, counter)
            if (epoch == 2) and (self.is_save):
                self.save(config.checkpoint_dir, counter)
            if (epoch == 3) and (self.is_save):
                self.save(config.checkpoint_dir, counter)
            if (epoch == 4) and (self.is_save):
                self.save(config.checkpoint_dir, counter)
            if (epoch == 5) and (self.is_save):
                self.save(config.checkpoint_dir, counter)



    @property
    def model_dir(self):
        return "{}_{}".format(
            self.dataset_name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = self.model_name
        checkpoint_dir = os.path.join(checkpoint_dir, self.ckpt_model_name, self.model_dir)
        #(checkpoint_dir = 'checkpoint/base_unet/origin_data_16')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        #(checkpoint_dir = ('checkpoint/base_unet/origin_data_16/UCell.model_%step')

        print(" [*] Save model in step [%2d/%2d]" %(step, (self.dataloader.len_images / self.batch_size) * self.epoch))


    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        if (self.config.new_check != "null"):
            checkpoint_dir = os.path.join(checkpoint_dir, self.ckpt_model_name, self.model_dir, self.config.new_check)
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, self.ckpt_model_name, self.model_dir)
           
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            print(" counter is {}".format(str(counter)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def make_csv(self):
        self.dataloader.prepare_test_images()
        test_images, test_origin_size, test_ids = self.dataloader.test_images()
        pred_images = []
        for i in range(int(len(test_ids) / self.batch_size)):
            #only for batch size 2, 4, 8, 16, 32
            pred_image = self.sess.run(self.labels_log, feed_dict = {
                                        self.inputs: test_images[i*self.batch_size:(i+1)*self.batch_size]})
            pred_images.extend(pred_image)
            #print(i)
            #print(int(len(test_ids) / self.batch_size))
            #print((len(test_ids) / self.batch_size) - 1)
            if i == (int((len(test_ids) / self.batch_size)) - 1):
                if int((len(test_ids) % self.batch_size)) != 0:
                    pred_image = self.sess.run(self.labels_log, feed_dict={
                    #self.inputs: test_images[(i * self.batch_size) + 1:((i + 1) * self.batch_size) + 1]})
                    self.inputs: test_images[len(test_images) - self.batch_size:len(test_images)]})
                    pred_images.extend(pred_image[self.batch_size - (len(test_images) % self.batch_size ): self.batch_size])
        print(" [*] making csv...")
        print(np.array(pred_images).shape)

        to_csv(pred_images, test_origin_size, test_ids, csv_name = self.csv_name, cut_off=self.config.cut_off)
