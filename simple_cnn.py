import tensorflow as tf
import tensorflow.contrib.slim as slim

def simple_cnn(inputs,
          num_classes=10,
          is_training=True,
          scope='simple_cnn',
        ):

  with tf.variable_scope(scope) as sc:

    net = slim.conv2d(inputs, 32, [3, 3], scope='conv1', 
        weights_initializer=tf.contrib.layers.xavier_initializer())
    net = slim.max_pool2d(net, [2, 2], scope='pool1')

    net = slim.conv2d(net, 64, [3, 3], scope='conv2', 
        weights_initializer=tf.contrib.layers.xavier_initializer())
    net = slim.max_pool2d(net, [2, 2], scope='pool2')

    net = slim.conv2d(net, 64, [3, 3], scope='conv3', 
        weights_initializer=tf.contrib.layers.xavier_initializer())
    
    net = slim.flatten(net)

    net = slim.fully_connected(net, 64, scope='fc1')

    net = slim.fully_connected(net, 10, scope='fc2')

    return net

def inference(inputs):
  return simple_cnn(inputs)
            

def loss(logits, labels, task_no):
  with tf.variable_scope('loss'):
    labels = tf.cast(labels, tf.int64) #type conversion
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy_loss', cross_entropy_mean)
    tf.add_to_collection('losses%d' %task_no, cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses%d' %task_no), name='total_loss')


def _add_loss_summaries(total_loss, task_no):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses%d' %task_no)
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))


  return loss_averages_op
