from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import argparse


def placeholder_inputs(batch_size, params):
    images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(batch_size,
                                                                     params.num_frames_per_clip,
                                                                     params.crop_size,
                                                                     params.crop_size,
                                                                     params.channels))
    labels_placeholder = tf.compat.v1.placeholder(tf.int64, shape=batch_size)
    return images_placeholder, labels_placeholder


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.compat.v1.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.compat.v1.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.compat.v1.nn.l2_loss(var) * wd
        tf.compat.v1.add_to_collection('losses', weight_decay)
    return var


def run_test(params):
    num_test_videos = len(list(open(params.test_list, 'r')))
    print("Number of test videos = {}".format(num_test_videos))

    images_placeholder, labels_placeholder = placeholder_inputs(params.batch_size * params.num_gpu, params)
    with tf.compat.v1.variable_scope('var_name') as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, params.num_classes], 0.04, 0.005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [params.num_classes], 0.04, 0.0),
        }

    logits = []
    for gpu_index in range(0, params.num_gpu):
        with tf.device('/gpu:%d' % gpu_index):
            logit = c3d_model.inference_c3d(images_placeholder[gpu_index * params.batch_size:(gpu_index + 1) * params.batch_size,:,:,:,:], 0.6, params.batch_size, weights, biases)
            logits.append(logit)
            
    logits = tf.concat(logits, 0)
    norm_score = tf.nn.softmax(logits)
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
  
    # Create a saver for writing training checkpoints.
    saver.restore(sess, params.model_name)
    # And then after everything is built, start the training loop.
    bufsize = 0
    write_file = open("predict_ret.txt", "w+", bufsize)
    next_start_pos = 0
    all_steps = int((num_test_videos - 1) / (params.batch_size * params.num_gpu) + 1)
    for step in xrange(all_steps):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        test_images, test_labels, next_start_pos, _, valid_len = input_data.read_clip_and_label(
            params.test_list,
            params.batch_size * params.num_gpu,
            start_pos=next_start_pos,
            crop_mean=params.crop_mean
        )
        predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
        )
        for i in range(0, valid_len):
            true_label = test_labels[i],
            top1_predicted_label = np.argmax(predict_score[i])
            # Write results: true label, class prob for true label, predicted label, class prob for predicted label
            write_file.write('{}, {}, {}, {}\n'.format(
                true_label[0],
                predict_score[i][true_label],
                top1_predicted_label,
                predict_score[i][top1_predicted_label]))
    
    write_file.close()
    print("done")


def main(_):
    parser = argparse.ArgumentParser(description='C3D Testing')
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num-classes', action='store', dest='num_classes', type=int, default=154, help='Number of classes')
    parser.add_argument('--crop-size', action='store', dest='crop_size', type=int, default=112, help='Crop size')
    parser.add_argument('--channels', action='store', dest='channels', type=int, default=3, help='Channels')
    parser.add_argument('--num-frames-per-clip', action='store', dest='num_frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--model-name', action='store', dest='model_name', type=str, default='checkpoints', help='Model saved directory')
    parser.add_argument('--num-gpu', action='store', dest='num_gpu', type=int, default=1, help='Number of gpus')
    parser.add_argument('--test-list', action='store', dest='test_list', type=str, help='Test list')
    parser.add_argument('--crop-mean', action='store', dest='crop_mean', type=str, help='Crop mean file')

    run_test(params=parser.parse_args())


if __name__ == '__main__':
    tf.compat.v1.app.run()
