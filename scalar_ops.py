from classifier import color_mask
import numpy as np
import tensorflow as tf

def accuracy(logits, labels):
  softmax = tf.nn.softmax(logits)
  argmax = tf.argmax(softmax, 3)
  
  shape = logits.get_shape().as_list()
  n = shape[3]
  
  one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
  equal_pixels = tf.reduce_sum(tf.to_float(color_mask(one_hot, labels)))
  total_pixels = reduce(lambda x, y: x * y, shape[:3])
  return equal_pixels / total_pixels


def pred_colors(logits):
  softmax = tf.nn.softmax(logits)
  argmax = tf.argmax(softmax, 3)

  shape = logits.get_shape().as_list()
  n = shape[3]

  one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
  #equal_pixels = tf.reduce_sum(tf.to_float(color_mask(one_hot, labels)))
  #total_pixels = reduce(lambda x, y: x * y, shape[:3])
  return one_hot


def loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  return tf.reduce_mean(cross_entropy, name='loss')


def loss_multigpu(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='loss')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def intersect_over_union(logits, labels, batch_size):
  logits = tf.argmax(logits, 3)
  labels = tf.argmax(labels, 3)
  pixels = tf.add(logits, labels)

  intersection = tf.reduce_sum(tf.to_float(tf.equal(pixels, 2)))
  union = tf.reduce_sum(tf.to_float(tf.greater_equal(pixels, 1)))

  return intersection / union


def imagewise_iou(logits, labels, batch_size):
  logits = tf.argmax(logits, 3)
  labels = tf.argmax(labels, 3)
  pixels = tf.add(logits, labels)

  intersection = tf.reduce_sum(tf.to_float(tf.equal(pixels, 2)), [1, 2])
  union = tf.reduce_sum(tf.to_float(tf.greater_equal(pixels, 1)), [1, 2])

  return tf.reduce_mean(tf.div(intersection, union))


def imagewise_iou_np(logits, labels, batch_size):
  average_iou = 0.0;
  for i in range(batch_size):
    label = labels[i, :, :, 0]
    logit = logits[i, :, :, 1]
    logit_zero_locs = np.int32(logit <= 0.5)  # was == 0.0
    label_zero_locs = np.int32(label == 255)
    union_m = logit_zero_locs + label_zero_locs
    union_tot = np.int32(union_m >= 1)
    intr_tot = np.int32(union_m == 2)
    iou = np.sum(intr_tot) / np.float(np.sum(union_tot))
    average_iou += iou

  return average_iou / batch_size