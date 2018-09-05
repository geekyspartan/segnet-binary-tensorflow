from inputs import inputs
from models import SegNetAutoencoder
from scalar_ops import accuracy, loss, intersect_over_union, imagewise_iou, imagewise_iou_np
from data_operations import *
import time

import classifier
import config
import tensorflow as tf
from scipy.misc import imread, imresize
import cv2
import scipy.io as sio
import numpy as np
import os


def testing_code(currentTestingItr, newDirPath, dirPath, images, labels, one_hot_labels, autoencoder, logits, FLAGS, trX, trY, valX, valY, teX, teY):
  num_channels = 3  
  
  os.chdir(dirPath + 'segnet/train_logs/')
  newDirPath = newDirPath + "/" + str(currentTestingItr + 1)
  os.mkdir(dirPath + 'segnet/train_logs/' + newDirPath)
  os.mkdir(dirPath + 'segnet/train_logs/' + newDirPath + "/train")
  os.mkdir(dirPath + 'segnet/train_logs/' + newDirPath + "/test")
  

  # tr_range = np.arange(trX.shape[0])
  # np.random.shuffle(tr_range)
  # trX = trX[tr_range[:budget]]
  # trY = trY[tr_range[:budget]]

  # TAKE SINGLE CHANNEL ENDS

  trX = trX.astype(np.float32)
  trY = trY.astype(np.float32)
  valX = valX.astype(np.float32)
  valY = valY.astype(np.float32)
  teX = teX.astype(np.float32)
  teY = teY.astype(np.float32)


  print "****************************"
  print trX.shape
  print trY.shape
  print valX.shape
  print valY.shape
  print teX.shape
  print teY.shape
  print "****************************"

  mode = 'all'

  #rootdir = './ckpts-explicit-'+mode+'-'

  folder = FLAGS.ckpt_dir;

  #accuracy_op = intersect_over_union(logits, one_hot_labels, FLAGS.batch)
  #accuracy_op = imagewise_iou(logits,one_hot_labels,FLAGS.batch)

  # logits = tf.slice(logits,[0,8,8,0],[-1,48,80,-1])

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
    max_val_accuracy = 0
    max_traing_accuracy = 0
    logits_value_for_mat = np.full((FLAGS.batch,48,80,2), 0.0)

    f = open(dirPath + 'segnet/train_logs/' + newDirPath + "/accuracy.txt",'w', 0)
    for step in range(FLAGS.steps):
      current_train_accuracy_log = []
      current_val_accuracy_log = []
      for i in range(int(trX.shape[0]/FLAGS.batch)):
        _, current_accuracy,curr_logits, current_loss = sess.run([train_step,accuracy_op,logits,loss_op],feed_dict={images:trX[i*FLAGS.batch:(i+1)*FLAGS.batch],labels:trY[i*FLAGS.batch:(i+1)*FLAGS.batch]})
        #current_accuracy = imagewise_iou_np(curr_logits,trY[i*FLAGS.batch:(i+1)*FLAGS.batch],FLAGS.batch)
        if max_traing_accuracy < current_accuracy:
          max_traing_accuracy = current_accuracy
          for count in range(0,FLAGS.batch):
            np_image = np.ones((48,80, 3),dtype=np.float)
            np_merge = np.ones((48 , 80*3 + 20, 3),dtype=np.float)
            for j in range(0, 48):
              for k in range(0, 80):
                logits_value_for_mat[count][j][k][0] = curr_logits[count][j][k][0]
                logits_value_for_mat[count][j][k][1] = curr_logits[count][j][k][1]
                if curr_logits[count][j][k][0] > curr_logits[count][j][k][1]:
                  np_image[j][k] = 0
                else:
                  np_image[j][k] = 255
            np_merge[: , 0 : 80] = trX[i * FLAGS.batch + count]
            np_merge[: , 80 : 80 + 10] = 224 
            np_merge[: , 80 + 10 : 80*2 + 10] = trY[i * FLAGS.batch + count]
            np_merge[: , 80*2 + 10 : 80*2 + 20] = 224 
            np_merge[: , 80*2 + 20: 80*3 + 20] = np_image
            cv2.imwrite(dirPath + 'segnet/train_logs/' + newDirPath + "/train/" + str(count) + "_training.png", np_merge)
          # sio.savemat(dirPath + 'segnet/logitst_mat', {"logits":logits_value_for_mat})
        current_train_accuracy_log.append(current_accuracy)
      f.write("Training accuracy: " + str(sum(current_train_accuracy_log)/len(current_train_accuracy_log)) + "\n")
      print "Training accuracy: " + str(sum(current_train_accuracy_log)/len(current_train_accuracy_log))

      for i in range(int(valX.shape[0]/FLAGS.batch)):
        current_accuracy,curr_logits = sess.run([accuracy_op,logits],feed_dict={images:valX[i*FLAGS.batch:(i+1)*FLAGS.batch],labels:valY[i*FLAGS.batch:(i+1)*FLAGS.batch]})
        #current_accuracy = imagewise_iou_np(curr_logits,valY[i*FLAGS.batch:(i+1)*FLAGS.batch],FLAGS.batch)
        current_val_accuracy_log.append(current_accuracy)
        # print str(i)+" Current validation accuracy: "+str(current_accuracy)
      print "Validation accuracy: " + str(sum(current_val_accuracy_log) / len(current_val_accuracy_log))

      mean_val_accuracy = sum(current_val_accuracy_log)/len(current_val_accuracy_log)
      #if step % FLAGS.summary_step == 0:
      #  summary_str = sess.run(summary)
      #  summary_writer.add_summary(summary_str, step)
      if mean_val_accuracy > max_val_accuracy: #and abs(mean_val_accuracy - current_val_accuracy) < 0.1:#step % FLAGS.batch == 0:
        max_val_accuracy = mean_val_accuracy
        test_accuracy_log = []
        # preds_log = np.zeros((teX.shape[0],720,960,32),dtype=np.float32)
        for i in range(int(teX.shape[0] / FLAGS.batch)):
          [curr_logits, current_accuracy] = sess.run([logits, accuracy_op],
                                                     feed_dict={images: teX[i * FLAGS.batch : (i + 1) * FLAGS.batch],
          
                                                                labels: teY[i * FLAGS.batch : (i + 1) * FLAGS.batch]})
          for count in range(0,FLAGS.batch):
            np_image = np.ones((48,80, 3),dtype=np.float)
            np_merge = np.ones((48 , 80*3 + 20, 3),dtype=np.float)
            for j in range(0, 48):
              for k in range(0, 80):
                if curr_logits[count][j][k][0] > curr_logits[count][j][k][1]:
                  np_image[j][k] = 0
                else:
                  np_image[j][k] = 255
            np_merge[: , 0 : 80] = teX[i * FLAGS.batch + count]
            np_merge[: , 80 : 80 + 10] = 224 
            np_merge[: , 80 + 10 : 80*2 + 10] = teY[i * FLAGS.batch + count]
            np_merge[: , 80*2 + 10 : 80*2 + 20] = 224 
            np_merge[: , 80*2 + 20: 80*3 + 20] = np_image
            cv2.imwrite(dirPath + 'segnet/train_logs/' + newDirPath + "/test/" + str(count) + "_testing.png", np_merge)
            # cv2.imwrite(dirPath + 'segnet/train_logs/' + str(i) + "_originalSegmentation.png", teY[i * FLAGS.batch + count])
            # cv2.imwrite(dirPath + 'segnet/train_logs/' + str(i) + "_original.png", teX[i * FLAGS.batch + count])
            # cv2.imwrite(dirPath + 'segnet/train_logs/' + str(i) + "_generatedSegmentation.png", np_image)

          # current_accuracy = imagewise_iou_np(curr_logits,teY[i*FLAGS.batch:(i+1)*FLAGS.batch],FLAGS.batch)
          test_accuracy_log.append(current_accuracy)
        print "Test accuracy: " + str(sum(test_accuracy_log) / len(test_accuracy_log))
        f.write("Test accuracy: " + str(sum(test_accuracy_log)/len(test_accuracy_log)) + "\n")
      else:
        f.write("Empty" + "\n")

        #min_loss = current_loss
        #if save:
        #  saver.save(sess, FLAGS.train_ckpt, global_step=global_step)
      print "Step done!"
    f.write("Final test accuracy: " + str(sum(test_accuracy_log) / len(test_accuracy_log)))
    print "Final test accuracy: " + str(sum(test_accuracy_log) / len(test_accuracy_log))
    # f.write(str(sum(test_accuracy_log)/len(test_accuracy_log)))
    f.close()
    return sum(test_accuracy_log) / len(test_accuracy_log)

if __name__ == "__main__":
  budgetFake = 500
  budgetReal = 0
  iterations = 50
  batchSize = 50
  averageAccuracy = 0.0
  totalTestingCount = 20
  dirPath = "/gpfs/home/anarora/"
  # dirPath = "/Users/anuragarora/Desktop/ppts/CSE523/"
  newDirPath = str(budgetFake) + "_Fake_Random_" + str(budgetReal) + "_Real_" + str(iterations) + "_count_" + str(totalTestingCount)  #_SubModular_TFTR_  #_Random_TFTR_
  os.mkdir(dirPath + 'segnet/train_logs/' + newDirPath)
  rootdir = 'rootdir'
  time_str = str(time.time())
  tf.app.flags.DEFINE_string('ckpt_dir', rootdir + time_str, 'Train checkpoint directory')
  #tf.app.flags.DEFINE_string('train', train_file, 'Train data')
  # tf.app.flags.DEFINE_string('train_ckpt', rootdir+time_str+'/model.ckpt', 'Train checkpoint file')
  #tf.app.flags.DEFINE_string('train_labels', train_labels_file, 'Train labels data')
  tf.app.flags.DEFINE_boolean('strided', True, 'Use strided convolutions and deconvolutions')
  # tf.app.flags.DEFINE_integer('summary_step', 10, 'Number of iterations before serializing log data')
  tf.app.flags.DEFINE_integer('batch', batchSize, 'Batch size')
  tf.app.flags.DEFINE_integer('steps', iterations, 'Number of training iterations') # was 2
  FLAGS = tf.app.flags.FLAGS
  #images, labels = inputs(FLAGS.batch, FLAGS.train, FLAGS.train_labels)
  images = tf.placeholder(tf.float32, [FLAGS.batch, 48,80,3]) # was time_Step, input_ve # swapped inpput vec and time step
  labels = tf.placeholder(tf.float32, [FLAGS.batch, 48,80,3]) # was 10

  one_hot_labels = classifier.one_hot(labels) # was labels

  autoencoder = SegNetAutoencoder(2, strided=FLAGS.strided)

  logits = autoencoder.inference(images) # was images

  realData = False
  images_dir_train = dirPath + 'segnet/data/yeast/originalData/originaltrainimages/'
  labels_dir_train = dirPath + 'segnet/data/yeast/originalData/originaltrainsegmCorrected/'

  extraData = True
  images_dir_train_extra = dirPath + 'segnet/data/yeast/fakeDataLarge/fakeimages/'
  labels_dir_train_extra = dirPath + 'segnet/data/yeast/fakeDataLarge/fakesegmCorrected/'

  images_dir_val = dirPath + 'segnet/data/yeast/valData/valimages/'
  labels_dir_val = dirPath + 'segnet/data/yeast/valData/valsegmCorrected/'

  images_dir_test = dirPath + 'segnet/data/yeast/testdata/testimages/'
  labels_dir_test = dirPath + 'segnet/data/yeast/testdata/testsegmCorrected/'

  totalStartTime = time.time()
  trX,valX,teX,trY,valY,teY, trExtraX, trExtraY = read_data(images_dir_train,labels_dir_train,images_dir_val,labels_dir_val,images_dir_test,labels_dir_test, images_dir_train_extra, labels_dir_train_extra, extraData, realData, budgetFake, budgetReal)
  totalStartTimeAfterRead = time.time()
  f = open(dirPath + 'segnet/train_logs/' + newDirPath + "/averageAccuracy.txt" ,'w', 0)
  for i in range(totalTestingCount):
      
      if(extraData):
          tr_range = np.arange(trExtraX.shape[0])
          np.random.shuffle(tr_range)
          trExtraX_temp = trExtraX[tr_range[:budget]]
          trExtraY_temp = trExtraY[tr_range[:budget]]

          if (realData):
              trX_temp = np.append(trX, trExtraX_temp, axis = 0)
              trY_temp = np.append(trY, trExtraY_temp, axis = 0)
          else:
              trX_temp = trExtraX_temp
              trY_temp = trExtraY_temp
      else:
          trX_temp = trX
          trY_temp = trY

      start = time.time()
      accu = testing_code(i, newDirPath, dirPath, images, labels, one_hot_labels, autoencoder, logits, FLAGS, trX_temp, trY_temp, valX, valY, teX, teY)
      
      f.write(i + " iteration accuracy: " + accu)
      f.write(i + " time: " + str((time.time() - start)))
      averageAccuracy = averageAccuracy + accu
      
  
  f.write("Final test accuracy: " + str(averageAccuracy / float(totalTestingCount)))
  f.write("Total time excluding read operation: " + str((time.time() - totalStartTimeAfterRead)))
  f.write("Total time including read operation: " + str((time.time() - totalStartTime)))
  f.close()




