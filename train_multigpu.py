from inputs import inputs
from models import SegNetAutoencoder
from scalar_ops import accuracy, loss_multigpu, intersect_over_union, imagewise_iou, imagewise_iou_np
#from tqdm import tqdm
from data_operations import *
import time

import scipy.io as sio

import shutil
import classifier
import config
import tensorflow as tf
import utils
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_iou

import sys

from scipy import misc

import numpy as np

import copy

import os

import re


def tower_loss(scope, images, labels,autoencoder):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits = autoencoder.inference(images)

  #one_hot_labels = classifier.one_hot(labels)

  logits = tf.slice(logits, [0, 8, 0, 0], [-1, 48, -1, -1])

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  #_ = autoencoder.lossseg(logits, labels)

  _ = loss_multigpu(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss, logits

def average_gradients(tower_grads):
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
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


num_channels = 3;

images_dir = '/home/hilab/Desktop/ali/submodularity_experiments/roadseg/701_StillsRaw_full/'
labels_dir = '/home/hilab/Desktop/ali/submodularity_experiments/roadseg/labels/labels/'


trX,valX,teX,trY,valY,teY = read_data(images_dir,labels_dir)

budget = 20


#trX, trY = submodular_choice(trX,trY,budget)


tr_range = np.arange(trX.shape[0])
np.random.shuffle(tr_range)

trX = trX[tr_range[:budget]]
trY = trY[tr_range[:budget]]




print trX.shape

# TAKE SINGLE CHANNEL ENDS

trX = trX.astype(np.float32)
trY = trY.astype(np.float32)
valX = valX.astype(np.float32)
valY = valY.astype(np.float32)
teX = teX.astype(np.float32)
teY = teY.astype(np.float32)


mode = 'all'

#rootdir = './ckpts-explicit-'+mode+'-'
time_str = str(int(time.time()))

rootdir = 'rootdir'

time_str = str(time.time())

tf.app.flags.DEFINE_string('ckpt_dir', rootdir+time_str, 'Train checkpoint directory')
#tf.app.flags.DEFINE_string('train', train_file, 'Train data')
tf.app.flags.DEFINE_string('train_ckpt', rootdir+time_str+'/model.ckpt', 'Train checkpoint file')
#tf.app.flags.DEFINE_string('train_labels', train_labels_file, 'Train labels data')
tf.app.flags.DEFINE_string('train_logs', './logs/train-explicit', 'Log directory')

tf.app.flags.DEFINE_boolean('strided', True, 'Use strided convolutions and deconvolutions')

tf.app.flags.DEFINE_integer('summary_step', 10, 'Number of iterations before serializing log data')
tf.app.flags.DEFINE_integer('batch', 1, 'Batch size')
tf.app.flags.DEFINE_integer('epochs', 10, 'Number of training iterations') # was 2

FLAGS = tf.app.flags.FLAGS

folder = FLAGS.ckpt_dir;

TOWER_NAME = 'tower'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
session_config = tf.ConfigProto(gpu_options=gpu_options) # allow_soft_placement=True

ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)


num_gpus = 1

#with tf.Session(config=session_config) as sess:


cnt = 0

# tf.Graph().as_default()

with tf.device('/cpu:0'):
  #colors = tf.cast(tf.stack(utils.colors_of_dataset(config.working_dataset)), tf.float32)
  images = tf.placeholder(tf.float32, [FLAGS.batch*num_gpus, 48, 80, 3])
  labels = tf.placeholder(tf.float32, [FLAGS.batch*num_gpus, 48, 80, 3])

  optimizer = tf.train.GradientDescentOptimizer(10e-3)
  autoencoder = SegNetAutoencoder(32, strided=FLAGS.strided)
  tower_grads = []
  accuracy_vals = []
  with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(num_gpus):
          with tf.device('/gpu:%d' % i):
              with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

                  image_batch, label_batch = images[i*FLAGS.batch:(i+1)*FLAGS.batch], labels[i*FLAGS.batch:(i+1)*FLAGS.batch]

                  one_hot_label_batch = classifier.one_hot(label_batch)

                  loss,logits = tower_loss(scope, image_batch, one_hot_label_batch,autoencoder)
                  tf.get_variable_scope().reuse_variables()
                  inference_op = autoencoder.inference(image_batch)
                  #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                  # Calculate the gradients for the batch of data on this CIFAR tower.
                  grads = optimizer.compute_gradients(loss)

                  # Keep track of the gradients across all towers.
                  tower_grads.append(grads)
                  #tower_logits.append(logits)
                  #tower_labels.append(one_hot_label_batch)

                  accuracy_vals.append(accuracy(logits, one_hot_label_batch))

                  pass
      accuracy_vals = tf.stack(accuracy_vals)
      accuracy_op = tf.reduce_mean(accuracy_vals)

      grads = average_gradients(tower_grads)
      train_op = optimizer.apply_gradients(grads)

      #logits = autoencoder.inference(images)

      #logits = tf.slice(logits, [0, 8, 0, 0], [-1, 720, -1, -1])

      #one_hot_labels = classifier.one_hot(labels)

      #accuracy_op = accuracy(logits,one_hot_labels)

      #train_op = apply_gradient_op

      ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

      init = tf.global_variables_initializer()

      sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False))
      sess.run(init)

      with sess:
      # if not ckpt:
      #   print('No checkpoint file found. Initializing...')
      #   global_step = 0
      #   #sess.run(init)
      #   tf.initialize_all_variables().run()
      # else:
      #   global_step = len(ckpt.all_model_checkpoint_paths) * FLAGS.steps
      #   ckpt_path = ckpt.model_checkpoint_path
      #   saver.restore(sess, ckpt_path)

      #summary = tf.merge_all_summaries()
      #summary_writer = tf.train.SummaryWriter(FLAGS.train_logs, sess.graph)
          max_val_accuracy = 0;
          for epoch in range(FLAGS.epochs):
              for step in range((trX.shape[0])/(FLAGS.batch*num_gpus)-1):
                start_time = time.time()
                current_accuracy, _, loss_value = sess.run([accuracy_op, train_op, loss],feed_dict={images:trX[step*FLAGS.batch*num_gpus:(step+1)*FLAGS.batch*num_gpus],labels:trY[step*FLAGS.batch*num_gpus:(step+1)*FLAGS.batch*num_gpus]})
                duration = time.time() - start_time
                #print str(step*FLAGS.batch*num_gpus) + ' to ' + str((step+1)*FLAGS.batch*num_gpus) + " " + str(duration) + " " + str(loss_value)
                #print current_accuracy
                start_time = time.time()
                _ = sess.run(inference_op,feed_dict={images:trX[step*FLAGS.batch*num_gpus:(step+1)*FLAGS.batch*num_gpus],labels:trY[step*FLAGS.batch*num_gpus:(step+1)*FLAGS.batch*num_gpus]})
                duration = time.time() - start_time 
                print duration

              current_val_accuracy_log = []
              for step in range((valX.shape[0]) / (FLAGS.batch * num_gpus)-1):
                  current_accuracy, curr_logits = sess.run([accuracy_op,logits], feed_dict={
                      images: valX[step * FLAGS.batch * num_gpus:(step + 1) * FLAGS.batch * num_gpus],
                      labels: valY[step * FLAGS.batch * num_gpus:(step + 1) * FLAGS.batch * num_gpus]})
                  current_val_accuracy_log.append(current_accuracy)
              mean_val_accuracy = sum(current_val_accuracy_log) / len(current_val_accuracy_log)
              print mean_val_accuracy
              if mean_val_accuracy > max_val_accuracy:
                  max_val_accuracy = mean_val_accuracy
                  test_accuracy_log = []
                  # preds_log = np.zeros((teX.shape[0],720,960,32),dtype=np.float32)
                  for step in range(int((teX.shape[0]) / (FLAGS.batch * num_gpus))-1):
                      [curr_logits, current_accuracy] = sess.run([logits, accuracy_op],
                                                                 feed_dict={
                                                                     images: teX[step * FLAGS.batch * num_gpus:(step + 1) * FLAGS.batch * num_gpus],
                                                                     labels: teY[step * FLAGS.batch * num_gpus:(step + 1) * FLAGS.batch * num_gpus]})
                      # current_accuracy = imagewise_iou_np(curr_logits,teY[i*FLAGS.batch:(i+1)*FLAGS.batch],FLAGS.batch)
                      test_accuracy_log.append(current_accuracy)
                  print "Done. Test accuracy: " + str(sum(test_accuracy_log) / len(test_accuracy_log))

print "Step done!"
print "Final test accuracy: " + str(sum(test_accuracy_log) / len(test_accuracy_log))
f = open('/home/hilab/Desktop/ali/submodularity_experiments/segnet/train_logs/'+str(budget)+"_"+time_str+'.txt','w')
f.write(str(sum(test_accuracy_log)/len(test_accuracy_log)))
f.close()



