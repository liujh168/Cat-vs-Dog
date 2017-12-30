import catdog_input, catdog_model
import tensorflow as tf
import numpy as np
import time

from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecords_name', 'train', """filename of tfrecords""")
tf.app.flags.DEFINE_string('max_step', 1000000, """max step of training""")

LEARNING_RATE = 0.001
IMAGE_WIDTH = catdog_input.IMAGE_WIDTH
IMAGE_HEIGHT = catdog_input.IMAGE_HEIGHT
IMAGE_CHANNEL = catdog_input.IMAGE_CHANNEL
NUM_CLASS = catdog_input.NUM_CLASS

NUM_EXAPLME_PER_EPOCH_FOR_TRAIN = catdog_input.NUM_EXAPLME_PER_EPOCH_FOR_TRAIN
NUM_EXAPLME_PER_EPOCH_FOR_TEST = catdog_input.NUM_EXAPLME_PER_EPOCH_FOR_TEST

BATCH_SIZE = catdog_input.BATCH_SIZE


def train():

    # global step is necessary for "StopAtStepHook"
    global_step = tf.train.get_or_create_global_step()

    with tf.device('/cpu:0'):
        images, labels = catdog_input.distorted_input(FLAGS.tfrecords_name, BATCH_SIZE)

    logits = catdog_model.inference(images)
    loss = catdog_model.loss(logits, labels)
    accuracy = catdog_model.accuracy(logits, labels)

    train_op = catdog_model.train(loss)

    class _LoggerHook(tf.train.SessionRunHook):
        """log"""

        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            # here return what you want to display when training
            # using run_values.result() to get the values
            return tf.train.SessionRunArgs([loss, accuracy])

        def after_run(self, run_context, run_values):

            # steps for displaying
            display_step = 1
            if self._step % display_step == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time
                loss, accuracy = run_values.results

                # examples used per seconds
                examples_per_sec = display_step * BATCH_SIZE / duration
                # the number of seconds for every batch
                sec_per_batch = float(duration / display_step)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), self._step, loss,
                                    examples_per_sec, sec_per_batch))

    # need to look up the documentation
    with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_step),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=False)) as sess:

        coord = tf.train.Coordinator()
        # open the image read queue
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while not sess.should_stop():
            sess.run(train_op)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()