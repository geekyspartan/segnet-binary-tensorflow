from inputs import inputs
from models import SegNetAutoencoder
from scalar_ops import accuracy, loss, intersect_over_union, imagewise_iou, imagewise_iou_np
#from tqdm import tqdm
from data_operations import *
import time


from utils_mnist import  load_mnist
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

num_channels = 3;

images_dir = '/home/hilab/Desktop/ali/submodularity_experiments/roadseg/701_StillsRaw_full/'
labels_dir = '/home/hilab/Desktop/ali/submodularity_experiments/roadseg/labels/labels/'



X = load_mnist("yeast")

#trX,valX,teX,trY,valY,teY = read_data(images_dir,labels_dir)

budget = 5


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
tf.app.flags.DEFINE_integer('batch', 2, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 10, 'Number of training iterations') # was 2

FLAGS = tf.app.flags.FLAGS

folder = FLAGS.ckpt_dir;


  #images, labels = inputs(FLAGS.batch, FLAGS.train, FLAGS.train_labels)
images = tf.placeholder(tf.float32, [FLAGS.batch, 48,80,3]) # was time_Step, input_ve # swapped inpput vec and time step
labels = tf.placeholder(tf.float32, [FLAGS.batch, 48,80,3]) # was 10

one_hot_labels = classifier.one_hot(labels) # was labels

autoencoder = SegNetAutoencoder(32, strided=FLAGS.strided)

logits = autoencoder.inference(images) # was images

#accuracy_op = intersect_over_union(logits, one_hot_labels, FLAGS.batch)
#accuracy_op = imagewise_iou(logits,one_hot_labels,FLAGS.batch)


logits = tf.slice(logits,[0,8,0,0],[-1,48,-1,-1])
accuracy_op = accuracy(logits,one_hot_labels)


loss_op = loss(logits, one_hot_labels)
#tf.summary.scalar('accuracy', accuracy_op)
#tf.summary.scalar(loss_op.op.name, loss_op)

optimizer = tf.train.AdamOptimizer(1e-04) # it was 9e-04,epsilon=1e-05 with smaller dataset
train_step = optimizer.minimize(loss_op)


tf.initialize_all_variables()

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
session_config = tf.ConfigProto(gpu_options=gpu_options) # allow_soft_placement=True

ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

with tf.Session(config=session_config) as sess:
  ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

  if not ckpt:
    print('No checkpoint file found. Initializing...')
    global_step = 0
    #sess.run(init)
    tf.initialize_all_variables().run()
  else:
    global_step = len(ckpt.all_model_checkpoint_paths) * FLAGS.steps
    ckpt_path = ckpt.model_checkpoint_path
    saver.restore(sess, ckpt_path)

  #summary = tf.merge_all_summaries()
  #summary_writer = tf.train.SummaryWriter(FLAGS.train_logs, sess.graph)
  max_val_accuracy = 0;

  for step in range(FLAGS.steps + 1):
    current_train_accuracy_log = []
    current_val_accuracy_log = []
    for i in range(int(trX.shape[0]/FLAGS.batch)):
      _, current_accuracy,curr_logits, current_loss = sess.run([train_step,accuracy_op,logits,loss_op],feed_dict={images:trX[i*FLAGS.batch:(i+1)*FLAGS.batch],labels:trY[i*FLAGS.batch:(i+1)*FLAGS.batch]})
      #current_accuracy = imagewise_iou_np(curr_logits,trY[i*FLAGS.batch:(i+1)*FLAGS.batch],FLAGS.batch)
      print str(i)+" Current training accuracy: "+str(current_accuracy)
      print str(i)+" Current loss: "+str(current_loss)
      current_train_accuracy_log.append(current_accuracy)
    for i in range(int(valX.shape[0]/FLAGS.batch)):
      current_accuracy,curr_logits = sess.run([accuracy_op,logits],feed_dict={images:valX[i*FLAGS.batch:(i+1)*FLAGS.batch],labels:valY[i*FLAGS.batch:(i+1)*FLAGS.batch]})
      #current_accuracy = imagewise_iou_np(curr_logits,valY[i*FLAGS.batch:(i+1)*FLAGS.batch],FLAGS.batch)
      current_val_accuracy_log.append(current_accuracy)
      print current_accuracy
    mean_val_accuracy = sum(current_val_accuracy_log)/len(current_val_accuracy_log)
    #if step % FLAGS.summary_step == 0:
    #  summary_str = sess.run(summary)
    #  summary_writer.add_summary(summary_str, step)
    #  summary_writer.flush()
    print mean_val_accuracy
    if mean_val_accuracy > max_val_accuracy: #and abs(mean_val_accuracy - current_val_accuracy) < 0.1:#step % FLAGS.batch == 0:
      max_val_accuracy = mean_val_accuracy
      test_accuracy_log = []
      # preds_log = np.zeros((teX.shape[0],720,960,32),dtype=np.float32)
      for i in range(int(teX.shape[0] / FLAGS.batch)):
        [curr_logits, current_accuracy] = sess.run([logits, accuracy_op],
                                                   feed_dict={images: teX[i * FLAGS.batch:(i + 1) * FLAGS.batch],
                                                              labels: teY[i * FLAGS.batch:(i + 1) * FLAGS.batch]})
        # current_accuracy = imagewise_iou_np(curr_logits,teY[i*FLAGS.batch:(i+1)*FLAGS.batch],FLAGS.batch)
        test_accuracy_log.append(current_accuracy)
      print "Done. Test accuracy: " + str(sum(test_accuracy_log) / len(test_accuracy_log))



      #min_loss = current_loss
      #if save:
      #  saver.save(sess, FLAGS.train_ckpt, global_step=global_step)
    print "Step done!"
  print "Final test accuracy: " + str(sum(test_accuracy_log) / len(test_accuracy_log))
  f = open('/home/hilab/Desktop/ali/submodularity_experiments/segnet/train_logs/'+str(budget)+"_"+time_str+'.txt','w')
  f.write(str(sum(test_accuracy_log)/len(test_accuracy_log)))
  f.close()



