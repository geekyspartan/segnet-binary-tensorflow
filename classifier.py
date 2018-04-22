import config
import tensorflow as tf
import utils

colors = tf.cast(tf.stack(utils.colors_of_dataset(config.working_dataset)), tf.float32) #/ 255

def color_mask(tensor, color):
  return tf.reduce_all(tf.equal(tensor, color), 3)

def one_hot(labels):
  color_tensors = tf.unstack(colors)
  channel_tensors = list(map(lambda color: color_mask(labels, color), color_tensors))
  one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
  return one_hot_labels

def rgb(logits):
  softmax = tf.nn.softmax(logits)
  # print "****************************-2"
  # print softmax #2, 64, 96, 32
  # print "****************************"
  argmax = tf.argmax(softmax, 3)
  # print "****************************-1"
  # print argmax #2, 64, 96
  # print "****************************"
  # print "****************************0"
  # print logits #2, 64, 96, 32
  # print "****************************"
  n = colors.get_shape().as_list()[0]
  one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
  # print "****************************1"
  # print str(one_hot.get_shape())
  # (2, 64, 96, 32)
  # print "****************************"
  # one_hot = tf.slice(one_hot,[0,8,8,0],[-1,48,80,-1])
  # (2, 48, 96, 32)   
  # print "****************************2"
  # print str(one_hot.get_shape())
  # print "****************************"
  one_hot_matrix = tf.reshape(one_hot, [-1, n])
  rgb_matrix = tf.matmul(one_hot_matrix, colors)
  # print "****************************3"
  # print logits
  # print str(one_hot_matrix.get_shape()) #(9216, 32)
  # print str(rgb_matrix.get_shape()) #(9216, 3)
  # print n #32
  # print colors #Tensor("Cast:0", shape=(32, 3), dtype=float32)
  # print "****************************"

  rgb_tensor = tf.reshape(rgb_matrix, [-1, 48, 80, 3])
  return tf.cast(rgb_tensor, tf.float32)
