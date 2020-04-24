#coding=utf-8
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.framework import ops

from datetime import datetime
import os.path
import re
import time

import cifar10_input
import simple_cnn

# Define parameters
FLAGS = tf.app.flags.FLAGS

# For cifar-10 training
tf.app.flags.DEFINE_string("train_dir", 'log/1machine_run',
                    'Directory where to write event logs and checkpoint.')

tf.app.flags.DEFINE_string("model", 'simple_cnn',
                    'Model to train')

tf.app.flags.DEFINE_integer('max_steps', 2000,
                    'Number of batches to run.')


def main(_):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  # fix random seed
  tf.set_random_seed(1234)

  # Global step
  global_step = tf.train.get_or_create_global_step()
  #global_step = tf.get_variable('global_step', [1], tf.int32, initializer=tf.constant_initializer(0), trainable=False)

  # Read shuffled input
  images, labels = cifar10_input.read_inputs()

  # Inference model
  logits = simple_cnn.inference(images)

  # Calculate loss
  loss = simple_cnn.loss(logits, labels, 0)

  # add regularization loss
  weight_decay=5e-4
  varlist = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
  lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in varlist if 'bias' not in v.name ]) \
            * weight_decay
  total_loss = loss + lossL2


  # Build a Graph that trains the model with one batch of examples and
  # updates the model parameters
  # Variables that affect learning rate.
  num_batches_per_epoch = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * cifar10_input.NUM_EPOCHS_PER_DECAY)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = simple_cnn._add_loss_summaries(total_loss, 0)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer()
    grads = opt.compute_gradients(total_loss,var_list=varlist)

  # Apply gradients.
  train_op = opt.apply_gradients(grads, global_step=global_step)

  # Create a saver.
  saver = tf.train.Saver(tf.global_variables())

  # Build the summary operation from the last tower summaries.
  summary_op = tf.summary.merge_all()

  # Initialize variables
  init_op = [tf.global_variables_initializer()]
  

  # Run session
  with tf.Session() as sess:
    sess.run(init_op)
    print("Session started!")

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in range(FLAGS.max_steps+1):
      start_time = time.time()
      _,loss_value = sess.run([train_op,loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 100 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 200 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      '''
      if step > 0 and step % 1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
      '''

    print("Session ended!")


if __name__ == "__main__":
  tf.app.run()
