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

class Solver(object):

  def __init__(self, train=True, common_params=None, solver_params=None, net_params=None, dataset_params=None):
    if common_params:
      self.device_id = int(common_params['gpus'])
      self.image_size = int(common_params['image_size'])
      self.height = self.image_size
      self.width = self.image_size
      self.batch_size = int(common_params['batch_size'])
      self.num_gpus = 1
    if solver_params:
      self.learning_rate = float(solver_params['learning_rate'])
      self.moment = float(solver_params['moment'])
      self.max_steps = int(solver_params['max_iterators'])
      self.train_dir = str(solver_params['train_dir'])
      self.lr_decay = float(solver_params['lr_decay'])
      self.decay_steps = int(solver_params['decay_steps'])
    self.train = train
    self.net = Net(train=train, common_params=common_params, net_params=net_params)
    self.dataset = DataSet(common_params=common_params, dataset_params=dataset_params)

  def construct_graph(self, scope):
    with tf.device('/gpu:' + str(self.device_id)):
      self.data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
      self.gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 313))
      self.prior_boost_nongray = tf.placeholder(tf.float32, (self.batch_size, int(self.height / 4), int(self.width / 4), 1))

      self.conv8_313 = self.net.inference(self.data_l)
      new_loss, g_loss = self.net.loss(scope, self.conv8_313, self.prior_boost_nongray, self.gt_ab_313)
      tf.summary.scalar('new_loss', new_loss)
      tf.summary.scalar('total_loss', g_loss)
    return new_loss, g_loss

  def train_model(self):
    with tf.device('/gpu:' + str(self.device_id)):
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           self.decay_steps, self.lr_decay, staircase=True)
      opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99)
      with tf.name_scope('gpu') as scope:
        new_loss, self.total_loss = self.construct_graph(scope)
        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
      grads = opt.compute_gradients(new_loss)

      self.summaries.append(tf.summary.scalar('learning_rate', learning_rate))

      for grad, var in grads:
        if grad is not None:
          self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

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
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      sess = tf.Session(config=config)
      sess.run(init)
      #saver1.restore(sess, './models/model.ckpt')
      #nilboy
      summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
      for step in xrange(self.max_steps):
        start_time = time.time()
        t1 = time.time()
        data_l, gt_ab_313, prior_boost_nongray = self.dataset.batch()
        t2 = time.time()
        _, loss_value = sess.run([train_op, self.total_loss], feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
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
          summary_str = sess.run(summary_op, feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
          summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0:
          checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
