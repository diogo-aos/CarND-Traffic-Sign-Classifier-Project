import tensorflow as tf
from tensorflow.contrib.layers import flatten

def model(x):
    'lenet'
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    input_depth = 3

    weights = {'wc1': tf.Variable(tf.truncated_normal([5, 5, input_depth, 6], mean=mu, stddev=sigma)),
               'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma)),
               'fc1': tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma)),
               'fc2': tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma)),
               'fc3': tf.Variable(tf.truncated_normal([84, n_classes], mean=mu, stddev=sigma))

              }
    biases = {'bc1': tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sigma)),
              'bc2': tf.Variable(tf.truncated_normal([16], mean=mu, stddev=sigma)),
              'fc1': tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sigma)),
              'fc2': tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sigma)),
              'fc3': tf.Variable(tf.truncated_normal([n_classes], mean=mu, stddev=sigma))
             }

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # TODO: Activation.
    pipeline = conv2d(x, weights['wc1'], biases['bc1'], strides=1, padding='VALID')

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pipeline = maxpool2d(pipeline, k=2)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    # TODO: Activation.
    pipeline = conv2d(pipeline, weights['wc2'], biases['bc2'], strides = 1, padding='VALID')

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pipeline = maxpool2d(pipeline, k=2)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    pipeline = tf.contrib.layers.flatten(pipeline)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # TODO: Activation.
    pipeline = fully_connect(pipeline, weights['fc1'], biases['fc1'])

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    # TODO: Activation.
    pipeline = fully_connect(pipeline, weights['fc2'], biases['fc2'])

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = fully_connect(pipeline, weights['fc3'], biases['fc3'], activate=False)

    return logits
