import tensorflow as tf
import numpy as np

learning_rate = 5e-3
batch_size = 16
n_epoch = 16
n_classes = 10


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.compat.v1.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.keras.initializers.glorot_normal())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.compat.v1.add_to_collection('weightdecay_losses', weight_decay)
  return var


with tf.compat.v1.variable_scope('var_name') as var_scope:
      weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
              'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
              'out': _variable_with_weight_decay('wout', [4096, 101], 0.0005)
              }
      biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
              'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
              'out': _variable_with_weight_decay('bout', [101], 0.000),
              }


export_dir = 'weights/conv3d_deepnetA_sport1m_iter_1900000_TF.model'

saver = tf.compat.v1.train.Saver(weights.values() + biases.values())
init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
sess.run(init)

saver.restore(sess, export_dir)

print()