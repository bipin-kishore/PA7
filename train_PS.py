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
tf.app.flags.DEFINE_string("train_dir", 'log/ps1worker3_run',
                    'Directory where to write event logs and checkpoint.')

tf.app.flags.DEFINE_string("model", 'simple_cnn',
                    'Model to train')

tf.app.flags.DEFINE_integer('max_steps', 2000,
                    'Number of batches to run.')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")

tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_bool("is_async",False,"Whether to use asynchronous PS")


def main(_):
  # fix random seed
  tf.compat.v1.set_random_seed(1234)

  # specify the cluster and create server
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  # Hint: function name 
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  # Hint: function name 
  server = tf.distribute.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()

  elif FLAGS.job_name == "worker":
    # Hint: function name 
    with tf.device(tf.compat.v1.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):

      # Global step
      global_step = tf.compat.v1.train.get_or_create_global_step()

      # Read shuffled input
      images, labels = cifar10_input.read_inputs()

      # Inference model
      logits = simple_cnn.inference(images)

      # Calculate loss
      loss = simple_cnn.loss(logits, labels, FLAGS.task_index)

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

      with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer()

        # For synchronous training. Hint: function name 
        opt = tf.compat.v1.train.SyncReplicasOptimizer(opt,
                                        replicas_to_aggregate=len(worker_hosts),
                                        total_num_replicas=len(worker_hosts),
                                        use_locking=True)

        # Compute gradients. Hint: see single machine version
        #[****Enter your code here****]
        grads = opt.compute_gradients(total_loss,var_list=varlist)

        # Apply gradients. Hint: see single machine version
      train_op = opt.apply_gradients(grads, global_step=global_step)#[****Enter your code here****]

    # worker 0 is chief worker
    is_chief = (FLAGS.task_index == 0)

    # For SyncReplicasOptimizer
    if FLAGS.is_async==False:
      sync_replicas_hook = [opt.make_session_run_hook(is_chief)]
    else:
      sync_replicas_hook = None

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge_all()

    # Initialize variables
    init_op = [tf.global_variables_initializer()]

    # create scaffold
    scaffold = tf.train.Scaffold(init_op=init_op,
                                summary_op=summary_op)


    # Run session
    with tf.train.MonitoredTrainingSession(master=server.target,
                                          is_chief=is_chief,
                                          scaffold=scaffold,
                                          hooks=sync_replicas_hook,
                                          save_summaries_steps=200) as sess:

      print("Session started!")

      summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

      step = 0
      step_print = 0
      while step <= FLAGS.max_steps:
        start_time = time.time()
        _,loss_value,next_step = sess.run([train_op,loss,global_step])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step == step_print and FLAGS.task_index == 0:
          num_examples_per_step = FLAGS.batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = duration

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))
          step_print += 100

        step = next_step
        while step > step_print:
          step_print += 100

      print("Session ended!")


if __name__ == "__main__":
  tf.compat.v1.app.run()
