from six.moves import xrange

import tensorflow as tf
import os
import time
import c3d_model
import input_data
import argparse


# flags = tf.compat.v1.flags
# flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
# flags.DEFINE_integer('batch_size', 10, 'Batch size.')
# flags.DEFINE_integer('num_classes', 154, 'Number of classes.')
# flags.DEFINE_integer('crop_size', 112, 'Crop size.')
# flags.DEFINE_integer('channels', 3, 'Number of channels.')
# flags.DEFINE_integer('num_frames_per_clip', 16, 'Number of frames per clip.')
# flags.DEFINE_string('save_model', 'c3d_general_model', 'Train model name')
# flags.DEFINE_string('model_save_dir', 'checkpoints', 'Model save dir')
# flags.DEFINE_float('moving_average_decay', 0.9999, 'Moving average decay')

# FLAGS = flags.FLAGS
# MOVING_AVERAGE_DECAY = 0.9999
# model_save_dir = './checkpoints'


def placeholder_inputs(batch_size, params):
    images_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(batch_size,
                                                                     params.num_frames_per_clip,
                                                                     params.crop_size,
                                                                     params.crop_size,
                                                                     params.channels))
    labels_placeholder = tf.compat.v1.placeholder(tf.int64, shape=batch_size)
    return images_placeholder, labels_placeholder


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss(name_scope, logit, labels):
    cross_entropy_mean = tf.reduce_mean(tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit))
    tf.compat.v1.summary.scalar(name_scope + '_cross_entropy', cross_entropy_mean)
    weight_decay_loss = tf.compat.v1.get_collection('weightdecay_losses')
    tf.compat.v1.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss))

    # Calculate the total loss for the current tower.
    total_loss = cross_entropy_mean + weight_decay_loss
    tf.compat.v1.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss))
    return total_loss


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.compat.v1.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.compat.v1.glorot_uniform_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.compat.v1.add_to_collection('weightdecay_losses', weight_decay)
    return var


def run_training(params):
    if not os.path.exists(params.model_save_dir):
        os.makedirs(params.model_save_dir)
        
    if hasattr(params, 'weights'):
        use_pretrained_model = True
    else:
        use_pretrained_model = False

    with tf.Graph().as_default():
        global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.compat.v1.constant_initializer(0), trainable=False)
        images_placeholder, labels_placeholder = placeholder_inputs(params.batch_size * params.num_gpu, params)

        tower_grads1 = []
        tower_grads2 = []
        logits = []
        opt_stable = tf.compat.v1.train.AdamOptimizer(1e-4)
        opt_finetuning = tf.compat.v1.train.AdamOptimizer(1e-3)

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
                'out': _variable_with_weight_decay('wout', [4096, params.num_classes], 0.0005)
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
                'out': _variable_with_weight_decay('bout', [params.num_classes], 0.000),
            }

        for gpu_index in range(0, params.num_gpu):
            with tf.device('/gpu:%d' % gpu_index):
                varlist2 = [weights['out'], biases['out']]
                varlist1 = list(set(list(weights.values()) + list(biases.values())) - set(varlist2))
                logit = c3d_model.inference_c3d(
                    images_placeholder[gpu_index * params.batch_size:(gpu_index + 1) * params.batch_size, :, :, :, :],
                    0.5,
                    params.batch_size,
                    weights,
                    biases
                )
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = tower_loss(
                    loss_name_scope,
                    logit,
                    labels_placeholder[gpu_index * params.batch_size:(gpu_index + 1) * params.batch_size]
                )
                grads1 = opt_stable.compute_gradients(loss, varlist1)
                grads2 = opt_finetuning.compute_gradients(loss, varlist2)
                tower_grads1.append(grads1)
                tower_grads2.append(grads2)
                logits.append(logit)

        logits = tf.concat(logits, 0)
        accuracy = tower_acc(logits, labels_placeholder)
        tf.compat.v1.summary.scalar('accuracy', accuracy)
        grads1 = average_gradients(tower_grads1)
        grads2 = average_gradients(tower_grads2)
        apply_gradient_op1 = opt_stable.apply_gradients(grads1)
        apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
        variable_averages = tf.compat.v1.train.ExponentialMovingAverage(params.moving_average_decay)
        variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())
        train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
        # null_op = tf.no_op()

        # Create a saver for writing training checkpoints.
        saver = tf.compat.v1.train.Saver(varlist1)
        init = tf.compat.v1.global_variables_initializer()
        
        saver_2 = tf.compat.v1.train.Saver(list(weights.values()) + list(biases.values()))

        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        if os.path.isfile(params.weights) and use_pretrained_model:
            saver.restore(sess, params.weights)

        # Create summary writer
        merged = tf.compat.v1.summary.merge_all()
        
        visual_logs_train = os.path.join(params.model_save_dir, 'visual_logs', 'train')
        visual_logs_test = os.path.join(params.model_save_dir, 'visual_logs', 'test')
        if not os.path.exists(visual_logs_train):
            os.makedirs(visual_logs_train)
        if not os.path.exists(visual_logs_test):
            os.makedirs(visual_logs_test)
        
        train_writer = tf.compat.v1.summary.FileWriter(visual_logs_train, sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(visual_logs_test, sess.graph)
        for step in xrange(params.max_steps):
            start_time = time.time()
            train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                filename=params.train_list,
                batch_size=params.batch_size * params.num_gpu,
                num_frames_per_clip=params.num_frames_per_clip,
                crop_size=params.crop_size,
                shuffle=True,
                crop_mean=params.crop_mean
            )
            sess.run(train_op, feed_dict={
                images_placeholder: train_images,
                labels_placeholder: train_labels
            })
            duration = time.time() - start_time
            print('Step %d: %.3f sec' % (step, duration))

            # Save a checkpoint and evaluate the model periodically.
            if step % 10 == 0 or (step + 1) == params.max_steps:
                saver_2.save(sess, os.path.join(params.model_save_dir), global_step=step)
                print('Training Data Eval:', end='')
                summary, acc = sess.run(
                    [merged, accuracy],
                    feed_dict={images_placeholder: train_images,
                               labels_placeholder: train_labels
                               })
                print(" accuracy: " + "{:.5f}, ".format(acc), end='')
                train_writer.add_summary(summary, step)
                print('Validation Data Eval:', end='')
                val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                    filename=params.test_list,
                    batch_size=params.batch_size * params.num_gpu,
                    num_frames_per_clip=params.num_frames_per_clip,
                    crop_size=params.crop_size,
                    shuffle=True,
                    crop_mean=params.crop_mean
                )
                summary, acc = sess.run(
                    [merged, accuracy],
                    feed_dict={
                        images_placeholder: val_images,
                        labels_placeholder: val_labels
                    })
                print(" accuracy: " + "{:.5f}".format(acc))
                test_writer.add_summary(summary, step)
        print("done")


def main(_):
    parser = argparse.ArgumentParser(description='C3D Training')
    parser.add_argument('--max-steps', action='store', dest='max_steps', type=int, default=5000, help='Number of steps to run trainer')
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num-classes', action='store', dest='num_classes', type=int, default=154, help='Number of classes')
    parser.add_argument('--crop-size', action='store', dest='crop_size', type=int, default=112, help='Crop size')
    parser.add_argument('--channels', action='store', dest='channels', type=int, default=3, help='Channels')
    parser.add_argument('--num-frames-per-clip', action='store', dest='num_frames_per_clip', type=int, default=16, help='Number of frames per clip')
    # parser.add_argument('--save-model', action='store', dest='save_model', type=str, default='c3d_model', help='Train model name')
    parser.add_argument('--model-save-dir', action='store', dest='model_save_dir', type=str, default='checkpoints', help='Model save directory')
    parser.add_argument('--moving-average-decay', action='store', dest='moving_average_decay', type=float, default=0.9999, help='Moving average decay')
    parser.add_argument('--num-gpu', action='store', dest='num_gpu', type=int, default=1, help='Number of gpus')
    parser.add_argument('--train-list', action='store', dest='train_list', type=str, help='Train list')
    parser.add_argument('--test-list', action='store', dest='test_list', type=str, help='Test list')
    parser.add_argument('--crop-mean', action='store', dest='crop_mean', type=str, help='Crop mean file')
    parser.add_argument('--pretrained-weights', action='store', dest='weights', type=str, help='Pretrained weights')
    
    run_training(params=parser.parse_args())
    

if __name__ == '__main__':
    tf.compat.v1.app.run()
