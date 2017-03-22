import uuid
from datetime import datetime as dt
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def equal_dicts (lhs, rhs):
    """Generic dictionary difference."""
    for key in lhs.keys():
          # auto-merge for missing key on right-hand-side.
        if key in rhs and lhs[key] == rhs[key]:
            continue
        else:
            return False
    for key in rhs.keys():
        if key not in lhs:
            return False
    return True


class ModelResult:
    def __init__(self, db=None, **params):
        self.db = db
        if params:
            self.init(params)

    def init(self, params):
        if 'model_name' not in params:
            raise Exception('model_name not in params')
        if 'learning_rate' not in params:
            raise Exception('learning_rate not in params')
        if 'epochs' not in params:
            raise Exception('epochs not in params')
        if 'batch_size' not in params:
            raise Exception('batch_size not in params')

        self.acc_counter = 0
        self.model_params = params
        self.accuracies = np.zeros((params['epochs'], 2), dtype=np.float32)
        self.uuid = str(uuid.uuid4())
        self.timestamp = dt.now()

    def check_db(self, db=None):
        if self.db:
            db = self.db
        if not db:
            raise ValueError('you must provide a file path for results database')

        try:
            with open(db, 'rb') as fp:
                results = pickle.load(fp)
        except (FileNotFoundError, EOFError) as e:
            results = {}

        in_params = params.copy()
        del in_params['epochs']
        for key, res in results.items():
            found_params = res['params'].copy()
            if 'epochs' in found_params:
                del found_params['epochs']

            if equal_dicts(in_params, found_params):
                self.accuracies = res['accuracies']
                self.results_key = key

                if params['epochs'] == res['params']['epochs']:
                    warnings.warn('model params already exist, diff epochs:{}'.format(res['params']['epochs']))
                else:
                    warnings.warn('model params already exist in results')

    def update(self, train_acc, val_acc, commit=False):
        self.accuracies[self.acc_counter, :] = train_acc, val_acc
        self.acc_counter += 1

        if commit:
            self.commit()

    def commit(self, to=None):
        if self.db:
            to = self.db
        if not to:
            raise ValueError('you must provide a file path for results database')

        try:
            with open(to, 'rb') as fp:
                results = pickle.load(fp)
        except (EOFError, FileNotFoundError) as e:
            print(e)
            results = {}

        if self.uuid not in results:
            results[self.uuid] = dict()
        results[self.uuid]['params'] = self.model_params
        results[self.uuid]['accuracies'] = self.accuracies
        results[self.uuid]['timestamp'] = self.timestamp
        with open(to, 'wb') as fp:
            pickle.dump(results, fp)

    def plot(self):
        i = self.acc_counter
        mp = self.model_params
        plt.clf()
        plt.plot(np.arange(i), self.accuracies[:i,0], label='train')
        plt.plot(np.arange(i), self.accuracies[:i,1], label='valid')

        plt.legend(loc='lower right')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('{},lr={},bs={},dr={}'.format(mp['model_name'], mp['learning_rate'], mp['batch_size'],
                                                mp['dropout_rate']))


def prelu(_x):
    #alphas = tf.get_variable('alpha', _x.get_shape()[-1],
    #                   initializer=tf.constant_initializer(0.1),
    #                    dtype=tf.float32)
    alphas = tf.Variable(0.1, _x.get_shape()[-1], dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

def conv2d(x, W, b, strides=1, padding='SAME', activation_func=tf.nn.relu):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return activation_func(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def fully_connect(x, W, b, activate=True, activation_func=tf.nn.relu):
    x = tf.matmul(x, W) + b
    if activate:
        return activation_func(x)
    return x
