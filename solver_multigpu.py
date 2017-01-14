from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from ops import *
from net import Net
from data import DataSet
import time
from datetime import datetime
import os
import sys

class SolverMultigpu(object):

  def __init__(self, train=True, common_params=None, solver_params=None, net_params=None, dataset_params=None):

    if common_params:
      self.gpus = [int(device) for device in str(common_params['gpus']).split(',')]
      self.image_size = int(common_params['image_size'])
      self.height = self.image_size
      self.width = self.image_size
      self.batch_size = int(common_params['batch_size']) / len(self.gpus)
    if solver_params:
      self.learning_rate = float(solver_params['learning_rate'])
      self.moment = float(solver_params['moment'])
      self.max_steps = int(solver_params['max_iterators'])
      self.train_dir = str(solver_params['train_dir'])
      self.lr_decay = float(solver_params['lr_decay'])
      self.decay_steps = int(solver_params['decay_steps'])
    self.tower_name = 'Tower'
    self.num_gpus = len(self.gpus)
    self.train = train
    self.net = Net(train=train, common_params=common_params, net_params=net_params)
    self.dataset = DataSet(common_params=common_params, dataset_params=dataset_params)
    self.placeholders=[]

  def construct_cpu_graph(self, scope):
    data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
    gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
    prior_boost_nongray = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))

    conv8_313 = self.net.inference(data_l)
    self.net.loss(scope, conv8_313, prior_boost_nongray, gt_ab_313)

  def construct_tower_gpu(self, scope):
    data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
    gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
    prior_boost_nongray = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))
    self.placeholders.append(data_l)
    self.placeholders.append(gt_ab_313)
    self.placeholders.append(prior_boost_nongray)

    conv8_313 = self.net.inference(data_l)
    new_loss, g_loss = self.net.loss(scope, conv8_313, prior_boost_nongray, gt_ab_313)
    tf.summary.scalar('new_loss', new_loss)
    tf.summary.scalar('total_loss', g_loss)
    return new_loss, g_loss

  def average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(0, grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads



  def train_model(self):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           self.decay_steps, self.lr_decay, staircase=True)
      opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99)

      with tf.name_scope('cpu_model') as scope:
        self.construct_cpu_graph(scope)
      tf.get_variable_scope().reuse_variables()
      tower_grads = []
      for i in self.gpus:
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (self.tower_name, i)) as scope:
            new_loss, self.total_loss = self.construct_tower_gpu(scope)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            grads = opt.compute_gradients(new_loss)
            tower_grads.append(grads)
      grads = self.average_gradients(tower_grads)

      self.summaries.append(tf.summary.scalar('learning_rate', learning_rate))
      
      for grad, var in grads:
        if grad is not None:
          self.summaries.append(
              tf.summary.histogram(var.op.name + '/gradients', grad))
      
      apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
      
      for var in tf.trainable_variables():
        self.summaries.append(tf.summary.histogram(var.op.name, var))

      variable_averages = tf.train.ExponentialMovingAverage(
          0.999, self.global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())


      train_op = tf.group(apply_gradient_op, variables_averages_op)

      saver = tf.train.Saver(write_version=1)
      saver1 = tf.train.Saver()
      summary_op = tf.summary.merge(self.summaries)
      init =  tf.global_variables_initializer()
      config=tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      sess.run(init)
      #saver1.restore(sess, self.pretrain_model)
      #nilboy
      summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
      for step in xrange(self.max_steps):
        start_time = time.time()
        t1 = time.time()
        feed_dict = {}
        np_feeds = []
        data_l, gt_ab_313, prior_boost_nongray = self.dataset.batch()
        for i in range(self.num_gpus):
          np_feeds.append(data_l[self.batch_size * i:self.batch_size * (i + 1),:,:,:])
          np_feeds.append(gt_ab_313[self.batch_size * i:self.batch_size * (i + 1),:,:,:])
          np_feeds.append(prior_boost_nongray[self.batch_size * i:self.batch_size * (i + 1),:,:,:])
        for i in range(len(self.placeholders)):
          feed_dict[self.placeholders[i]] = np_feeds[i]
        t2 = time.time()
        _, loss_value = sess.run([train_op, self.total_loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        t3 = time.time()
        print('io: ' + str(t2 - t1) + '; compute: ' + str(t3 - t2))
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 1 == 0:
          num_examples_per_step = self.batch_size * self.num_gpus
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration / self.num_gpus

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))
        
        if step % 10 == 0:
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0:
          checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
