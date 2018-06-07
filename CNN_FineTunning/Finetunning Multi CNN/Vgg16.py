"""
This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all 
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Following my blogpost at:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of classes.
The structure of this script is strongly inspired by the fast.ai Deep Learning
class by Jeremy Howard and Rachel Thomas, especially their vgg16 finetuning
script:  
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same folder: 
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/  

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np

class Vgg16(object):

  def __init__(self, x, keep_prob, num_classes, skip_layer,
               weights_path = 'DEFAULT'):

    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer

    if weights_path == 'DEFAULT':
      self.WEIGHTS_PATH = 'vgg16.npy'#'bvlc_alexnet.npy'#''vgg16.npy'
    else:
      self.WEIGHTS_PATH = weights_path

    # Call the create function to build the computational graph of Vgg16
    self.create()

  def create(self):
    #################################
    # conv1_1
    with tf.variable_scope('conv1_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(self.X, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope.name)

    # conv1_2
    with tf.variable_scope('conv1_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope.name)

    # conv2_2
    with tf.variable_scope('conv2_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3_1
    with tf.variable_scope('conv3_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope.name)

    # conv3_2
    with tf.variable_scope('conv3_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope.name)

    # conv3_3
    with tf.variable_scope('conv3_3') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope.name)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4_1
    with tf.variable_scope('conv4_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope.name)

    # conv4_2
    with tf.variable_scope('conv4_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope.name)

    # conv4_3
    with tf.variable_scope('conv4_3') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope.name)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # conv5_1
    with tf.variable_scope('conv5_1') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope.name)

    # conv5_2
    with tf.variable_scope('conv5_2') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope.name)

    # conv5_3
    with tf.variable_scope('conv5_3') as scope:
        kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope.name)

    # pool5
    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    # fc6
    with tf.variable_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.get_variable('weights', initializer=tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1))
        fc6b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        self.fc6 = tf.nn.dropout(fc6, self.KEEP_PROB)

    # fc7
    with tf.variable_scope('fc7') as scope:
        fc7w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1))
        fc7b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[4096], dtype=tf.float32))
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        self.fc7 = tf.nn.dropout(fc7, self.KEEP_PROB)

    # fc8
    with tf.variable_scope('fc8') as scope:
        fc8w = tf.get_variable('weights', initializer=tf.truncated_normal([4096, self.NUM_CLASSES], dtype=tf.float32, stddev=1e-1))
        fc8b = tf.get_variable('biases', initializer=tf.constant(1.0, shape=[self.NUM_CLASSES], dtype=tf.float32))
        self.fc8 = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)

    #return self.score
    #################################


  def load_initial_weights(self, session):
    """
    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
    as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
    dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
    need a special load function
    """

    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

      # Check if the layer is one of the layers that should be reinitialized
      if op_name not in self.SKIP_LAYER:


       #with tf.get_variable_scope().reuse_variables():# tf.get_variable_scope().reuse_variables()
          with tf.variable_scope(str(op_name), reuse = True):
            print('op_name done',op_name)

            # Loop over list of weights/biases and assign them to their corresponding tf variable
            for data in weights_dict[op_name]:


                # Biases
                if len(data.shape) == 1:

                    var = tf.get_variable('biases', trainable = False)
                    session.run(var.assign(data))

                # Weights
                else:

                    var = tf.get_variable('weights', trainable = False)
                    session.run(var.assign(data))
  ##################################################################
  def load_original_weights2(self, session, skip_layers=[]):
        weights_dict = np.load('vgg16.npy', encoding='bytes').item()

        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue

            if op_name == 'fc8' and self.NUM_CLASSES != 1000:
                continue

            with tf.variable_scope(op_name, reuse=True):
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases')
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights')
                        session.run(var.assign(data))
  ############################################################################

  def load_original_weights(self, session, skip_layers=[]):
      weights = np.load('vgg16_weights.npz')
      keys = sorted(weights.keys())

      for i, name in enumerate(keys):
          parts = name.split('_')
          layer = '_'.join(parts[:-1])

          if layer in self.SKIP_LAYER:# skip_layers:
                continue

          #if layer == 'fc8' or and self.NUM_CLASSES != 1000:
          #    continue

          with tf.variable_scope(layer, reuse=True):
              if parts[-1] == 'W':
                  var = tf.get_variable('weights')
                  session.run(var.assign(weights[name]))
              elif parts[-1] == 'b':
                  var = tf.get_variable('biases')
                  session.run(var.assign(weights[name]))

    