import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import flatten


def model(x):
    'DN2'
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0.0
    sigma = 0.005
    dropout = tf.placeholder(tf.float32, name='keep_prob')

    with tf.variable_scope('model'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(mu, sigma),
                            biases_initializer=tf.constant_initializer(0.1),
                            activation_fn=prelu):
            with tf.variable_scope('conv'):

                net = slim.conv2d(x, 6, [5, 5], padding='VALID')
                net = slim.conv2d(net, 12, [5, 5], padding='VALID')

                net = slim.conv2d(net, 20, [5, 5], padding='SAME')
                net = slim.conv2d(net, 28, [5, 5], padding='SAME')
                net = slim.max_pool2d(net, [2,2], 2)
                net = slim.conv2d(net, 36, [5, 5], padding='VALID')

            net = tf.contrib.layers.flatten(net)
            print(net.get_shape())
            with tf.variable_scope('fc'):
                net = slim.fully_connected(net, 4096)
                net = slim.dropout(net, dropout)
                net = slim.fully_connected(net, 2048)
                net = slim.dropout(net, dropout)
                net = slim.fully_connected(net, 1024)
                net = slim.dropout(net, dropout)
                net = slim.fully_connected(net, 512)
                net = slim.dropout(net, dropout)
                net = slim.fully_connected(net, 120)
                net = slim.fully_connected(net, 84)
                net = slim.fully_connected(net, 43, activation_fn=None, scope='logits')

    return net