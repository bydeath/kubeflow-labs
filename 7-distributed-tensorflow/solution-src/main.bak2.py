# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import ast
import json

import collections
import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """

        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming de60000pth == 1)
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , \
                   np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_data_sets(train_dir,
                   reshape=True,
                   validation_size=2000):
    trainfile = os.path.join(train_dir, "mnist_train.csv")
    testfile = os.path.join(train_dir, "mnist_test.csv")
    train_images = np.array([], dtype=np.uint8)
    train_labels = np.array([], dtype=np.uint8)
    test_images = np.array([], dtype=np.uint8)
    test_labels = np.array([], dtype=np.uint8)

    count = 0
    with open(trainfile) as f:
        for line in f.readlines():
            count+= 1
            line = line.strip()
            line = line.split(",")
            line = [int(x) for x in line]
            one_rray = np.array(line[1:], dtype=np.uint8)
            train_images = np.hstack((train_images, one_rray))
            train_labels = np.hstack((train_labels, np.array(line[0], dtype=np.uint8)))
            if count % 10000 == 0:
                print("trainfiles line number: %s " % str(count))
            if count == 20000:
                break
    train_images = train_images.reshape(20000, 28*28)
    train_labels = train_labels.reshape(20000, 1)
    train_labels = dense_to_one_hot(train_labels, 10)

    count = 0
    with open(testfile) as f:
        for line in f.readlines():
            count += 1
            line = line.strip()
            line = line.split(",")
            line = [int(x) for x in line]
            one_rray = np.array(line[1:], dtype=np.uint8)
            test_images = np.hstack((test_images, one_rray))
            test_labels = np.hstack((test_labels, np.array(line[0], dtype=np.uint8)))
            if count % 10000 == 0:
                print("testfiles line number: %s " % str(count))
    test_images = test_images.reshape(10000, 28*28)
    test_labels = test_labels.reshape(10000, 1)
    test_labels = dense_to_one_hot(test_labels, 10)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, reshape=reshape)
    test = DataSet(test_images, test_labels, reshape=reshape)


    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return Datasets(train=train, validation=validation, test=test)

def train():
  tf_config_json = os.environ.get("TF_CONFIG", "{}")
  tf_config = json.loads(tf_config_json)

  task = tf_config.get("task", {})
  cluster_spec = tf_config.get("cluster", {})
  cluster_spec_object = tf.train.ClusterSpec(cluster_spec)
  job_name = task["type"]
  task_id = task["index"]
  server_def = tf.train.ServerDef(
      cluster=cluster_spec_object.as_cluster_def(),
      protocol="grpc",
      job_name=job_name,
      task_index=task_id)
  server = tf.train.Server(server_def)

  is_chief = (job_name == 'master')

  # Import data
  mnist = read_data_sets(FLAGS.data_dir)

  
  # Create a multilayer model.


  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % task_id,
    cluster=cluster_spec)):

    # count the number of updates
    global_step = tf.get_variable(
      'global_step',
      [],
      initializer = tf.constant_initializer(0),
      trainable = False)

    # Input placeholders
    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, 784], name='x-input')
      y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('input_reshape'):
      image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
      tf.summary.image('input', image_shaped_input, 10)

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
      """Create a weight variable with appropriate initialization."""
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      """Create a bias variable with appropriate initialization."""
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
      """Reusable code for making a simple neural net layer.

      It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
      It also sets up name scoping so that the resultant graph is easy to read,
      and adds a number of summary ops.
      """
      # Adding a name scope ensures logical grouping of the layers in the graph.
      with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
          weights = weight_variable([input_dim, output_dim])
          variable_summaries(weights)
        with tf.name_scope('biases'):
          biases = bias_variable([output_dim])
          variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
          preactivate = tf.matmul(input_tensor, weights) + biases
          tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.summary.scalar('dropout_keep_probability', keep_prob)
      dropped = tf.nn.dropout(hidden1, keep_prob)

    # Do not apply softmax activation yet, see below.
    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
      # The raw formulation of cross-entropy,
      #
      # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
      #                               reduction_indices=[1]))
      #
      # can be numerically unstable.
      #
      # So here we use tf.nn.softmax_cross_entropy_with_logits on the
      # raw outputs of the nn_layer above, and then average across
      # the batch.
      diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
      with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
      train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
          cross_entropy)

    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()  

    init_op = tf.global_variables_initializer()

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}



  sv = tf.train.Supervisor(is_chief=is_chief,
						global_step=global_step,
						init_op=init_op,
						logdir=FLAGS.logdir)

  with sv.prepare_or_wait_for_session(server.target) as sess:  
    train_writer = tf.summary.FileWriter(FLAGS.logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.logdir + '/test')
    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries

    for i in range(FLAGS.max_steps):
      if i % 10 == 0:  # Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
      else:  # Record train set summaries, and train
        if i % 100 == 99:  # Record execution stats
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          summary, _ = sess.run([merged, train_step],
                                feed_dict=feed_dict(True),
                                options=run_options,
                                run_metadata=run_metadata)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
          train_writer.add_summary(summary, i)
          print('Adding run metadata for', i)
        else:  # Record a summary
          summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
          train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()


def main(_):
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/'),
                           'datanfs'),
      help='Directory for storing input data')
  parser.add_argument(
      '--logdir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/'),
                           'datanfs/logs'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)