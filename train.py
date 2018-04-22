from inputs import inputs
from scalar_ops import accuracy, loss
from tqdm import tqdm

import classifier
import config
import tensorflow as tf
import utils

dirPath = "/gpfs/home/anarora/"
# dirPath = "/Users/anuragarora/Desktop/ppts/CSE523/"

root_path = dirPath + 'segnet/'

train_file, train_labels_file = utils.get_training_set(config.working_dataset)

train_file = root_path + train_file
train_labels_file = root_path + train_labels_file



tf.app.flags.DEFINE_string('ckpt_dir', './ckpts', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('train', train_file, 'Train data')
tf.app.flags.DEFINE_string('train_ckpt', './ckpts/model.ckpt', 'Train checkpoint file')
tf.app.flags.DEFINE_string('train_labels', train_labels_file, 'Train labels data')
tf.app.flags.DEFINE_string('train_logs', './logs/train', 'Log directory')

tf.app.flags.DEFINE_integer('summary_step', 10, 'Number of iterations before serializing log data')
tf.app.flags.DEFINE_integer('batch', 12, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 10000, 'Number of training iterations')

FLAGS = tf.app.flags.FLAGS

def train():
  images, labels = inputs(FLAGS.batch, FLAGS.train, FLAGS.train_labels)
  tf.summary.image('labels', labels)
  one_hot_labels = classifier.one_hot(labels)

  autoencoder = utils.get_autoencoder(config.autoencoder, config.working_dataset, config.strided)
  logits = autoencoder.inference(images)

  accuracy_op = accuracy(logits, one_hot_labels)
  loss_op = loss(logits, one_hot_labels)
  tf.summary.scalar('accuracy', accuracy_op)
  tf.summary.scalar(loss_op.op.name, loss_op)

  optimizer = tf.train.AdamOptimizer(1e-04)
  train_step = optimizer.minimize(loss_op)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_memory_fraction)
  session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

  ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

  tf.initialize_all_variables()
  with tf.Session(config=session_config) as sess:
    tf.initialize_all_variables()
    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

    if not ckpt:
      print('No checkpoint file found. Initializing...')
      global_step = 0
      sess.run(init)
    else:
      global_step = len(ckpt.all_model_checkpoint_paths) * FLAGS.steps
      ckpt_path = ckpt.model_checkpoint_path
      saver.restore(sess, ckpt_path)

    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.train_logs, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in tqdm(range(FLAGS.steps + 1)):
      sess.run(train_step)

      if step % FLAGS.summary_step == 0:
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      if step % FLAGS.batch == 0:
        saver.save(sess, FLAGS.train_ckpt, global_step=global_step)

    coord.request_stop()
    coord.join(threads)

def main(argv=None):
  utils.restore_logs(FLAGS.train_logs)
  train()

if __name__ == '__main__':
  tf.app.run()
