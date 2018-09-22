from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
nilboy_weight_decay = 0.001
def _variable(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the Variable
    shape: list of ints
    initializer: initializer of Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with truncated normal distribution
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable 
    shape: list of ints
    stddev: standard devision of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight 
    decay is not added for this Variable.

 Returns:
    Variable Tensor 
  """
  var = _variable(name, shape,
    tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(scope, input, kernel_size, stride=1, dilation=1, relu=True, wd=nilboy_weight_decay):
  name = scope
  with tf.variable_scope(scope) as scope:
    kernel = _variable_with_weight_decay('weights', 
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    if dilation == 1:
      conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
    else:
      conv = tf.nn.atrous_conv2d(input, kernel, dilation, padding='SAME')
    biases = _variable('biases', kernel_size[3:], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    if relu:
      conv1 = tf.nn.relu(bias)
    else:
      conv1 = bias
  return conv1

def deconv2d(scope, input, kernel_size, stride=1, wd=nilboy_weight_decay):
  """convolutional layer

  Args:
    input: 4-D tensor [batch_size, height, width, depth]
    scope: variable_scope name 
    kernel_size: [k_height, k_width, in_channel, out_channel]
    stride: int32
  Return:
    output: 4-D tensor [batch_size, height * stride, width * stride, out_channel]
  """
  pad_size = int((kernel_size[0] - 1)/2)
  #input = tf.pad(input, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")
  batch_size, height, width, in_channel = [int(i) for i in input.get_shape()]
  out_channel = kernel_size[3] 
  kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
  output_shape = [batch_size, height * stride, width * stride, out_channel]
  with tf.variable_scope(scope) as scope:
    kernel = _variable_with_weight_decay('weights', 
                                    shape=kernel_size,
                                    stddev=5e-2,
                                    wd=wd)
    deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='SAME')

    biases = _variable('biases', (out_channel), tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(deconv, biases)
    deconv1 = tf.nn.relu(bias)

  return deconv1

def batch_norm(scope, x, train=True, reuse=False):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)

