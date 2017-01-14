from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from ops import *
from data import DataSet
import time
from datetime import datetime
import os
import sys

class Net(object):

  def __init__(self, train=True, common_params=None, net_params=None):
    self.train = train
    self.weight_decay = 0.0
    if common_params:
      gpu_nums = len(str(common_params['gpus']).split(','))
      self.batch_size = int(int(common_params['batch_size'])/gpu_nums)
    if net_params:
      self.weight_decay = float(net_params['weight_decay'])

  def inference(self, data_l):
    #conv1
    conv_num = 1

    temp_conv = conv2d('conv' + str(conv_num), data_l, [3, 3, 1, 64], stride=1, wd=self.weight_decay)
    conv_num += 1

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 64], stride=2, wd=self.weight_decay)
    conv_num += 1

    #self.nilboy = temp_conv

    temp_conv = batch_norm('bn_1', temp_conv,train=self.train)
    #conv2
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 64, 128], stride=1, wd=self.weight_decay)
    conv_num += 1
    
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 128], stride=2, wd=self.weight_decay)
    conv_num += 1

    temp_conv = batch_norm('bn_2', temp_conv,train=self.train)
    #conv3
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 128, 256], stride=1, wd=self.weight_decay)
    conv_num += 1
    
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
    conv_num += 1    

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=2, wd=self.weight_decay)
    conv_num += 1

    temp_conv = batch_norm('bn_3', temp_conv, train=self.train)
    #conv4
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 512], stride=1, wd=self.weight_decay)
    conv_num += 1
    
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
    conv_num += 1

    
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
    conv_num += 1

    temp_conv = batch_norm('bn_4', temp_conv,train=self.train)

    #conv5
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
    conv_num += 1    



    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
    conv_num += 1

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
    conv_num += 1

    temp_conv = batch_norm('bn_5', temp_conv,train=self.train)
    #conv6
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
    conv_num += 1    

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
    conv_num += 1

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, dilation=2, wd=self.weight_decay)
    conv_num += 1    

    temp_conv = batch_norm('bn_6', temp_conv,train=self.train)    
    #conv7
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
    conv_num += 1

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
    conv_num += 1

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 512, 512], stride=1, wd=self.weight_decay)
    conv_num += 1

    temp_conv = batch_norm('bn_7', temp_conv,train=self.train)
    #conv8
    temp_conv = deconv2d('conv' + str(conv_num), temp_conv, [4, 4, 512, 256], stride=2, wd=self.weight_decay)
    conv_num += 1    

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
    conv_num += 1

    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [3, 3, 256, 256], stride=1, wd=self.weight_decay)
    conv_num += 1

    #Unary prediction
    temp_conv = conv2d('conv' + str(conv_num), temp_conv, [1, 1, 256, 313], stride=1, relu=False, wd=self.weight_decay)
    conv_num += 1

    conv8_313 = temp_conv
    return conv8_313
  def loss(self, scope, conv8_313, prior_boost_nongray, gt_ab_313):
    
    flat_conv8_313 = tf.reshape(conv8_313, [-1, 313])
    flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1,313])
    g_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(flat_conv8_313, flat_gt_ab_313)) / (self.batch_size)
    
    tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses', scope=scope)))
    #
    dl2c = tf.gradients(g_loss, conv8_313)
    dl2c = tf.stop_gradient(dl2c)
    #
    new_loss = tf.reduce_sum(dl2c * conv8_313 * prior_boost_nongray) + tf.add_n(tf.get_collection('losses', scope=scope))
    return new_loss, g_loss
