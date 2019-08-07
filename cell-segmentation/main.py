# original code from https://github.com/jfpuget/DSB_2018.git
# reference source: 
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/54426#latest-482147
# https://www.kaggle.com/c/data-science-bowl-2018/discussion/54742#latest-322587


import os
import pprint

import scipy.misc
import numpy as np

from model import UCell

pp = pprint.PrettyPrinter()

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_float("net_init_stddev", 0.01, "u_net_init_stddev")
flags.DEFINE_float("net_bn_decay", 0.9,"u_net_bn_decay")
flags.DEFINE_float("net_bn_epsilon", 0.001,"unet_bn_epsilon")

flags.DEFINE_float("net_dropout_keep", 0.9, "u_net_dropout_keep")
flags.DEFINE_integer("unet_base_feature", 32, "unet_base_feature number!")
flags.DEFINE_integer("unet_step_size", 4, "the # of step in unet!")
flags.DEFINE_boolean("unet_weight", False, "true: use unet weight in loss function, false: not use")

   
flags.DEFINE_integer("opt_decay_steps", 1, "optimizer_decay_steps")
flags.DEFINE_float("opt_decay_rate", 0.98, "optimizer_decay_rate")
flags.DEFINE_float("lr_decay", 1.0, "decay of learning rate")
flags.DEFINE_integer("pre_erosion_iter", 1, "pre_erosion_iter")
    
flags.DEFINE_integer("epoch", 8, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 8, "The size of batch images [16]")

flags.DEFINE_integer("input_height", 228, "The size of image to use")
flags.DEFINE_integer("input_width", 228, "The size of image to use ")
flags.DEFINE_integer("img_channel", 3, "The size of image to use ")
flags.DEFINE_integer("label_channel", 1, "The size of image to use ")
flags.DEFINE_integer("aug_num", 8, "The number of augmented images")
flags.DEFINE_float("gan_loss_weight", 0.2, "the weight of gan_loss")
flags.DEFINE_float("cut_off", 0.5, "rate of cutoff")


flags.DEFINE_string("new_check", "null", "name of train checkpoint dir")
flags.DEFINE_string("optimizer", "adam", "name of optimizer")
flags.DEFINE_string("model_name", "UCell.model", "name of model")
flags.DEFINE_string("ckpt_model_name", "base_unet_init", "name of dir of checkpoint#2")
flags.DEFINE_string("dataset_name", "origin_data", "name of dataset")
flags.DEFINE_string("train_dir", "stage1_dataset/", "The dir of train_data")
flags.DEFINE_string("test_dir", "stage2_dataset/stage2_test_final/", "The dir of test_data")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("csv_name", "new_test.csv", "base csv name!")

flags.DEFINE_boolean("is_save", False, "true for saving model in epoch 4, false for not saving model")
flags.DEFINE_boolean("is_training", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")


FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(os.path.join(FLAGS.checkpoint_dir,FLAGS.ckpt_model_name)):
        os.makedirs(os.path.join(FLAGS.checkpoint_dir,FLAGS.ckpt_model_name))
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        seg_model = UCell(sess, FLAGS)

        if FLAGS.is_training:
            seg_model.train(FLAGS)
        else:
            if not seg_model.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        seg_model.make_csv()

        #OPTION = 1
        #visualize(sess, dcgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
