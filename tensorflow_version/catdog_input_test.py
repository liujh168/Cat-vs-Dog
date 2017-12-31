import tensorflow as tf

from matplotlib import pyplot as plt
import catdog_input


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tfrecords_file_name', 'test', """filename of tfrecords""")


if __name__ == '__main__':

    images, labels = catdog_input.distorted_input(FLAGS.tfrecords_file_name, batch_size=4)

    fig = plt.figure()
    a = fig.add_subplot(221)
    b = fig.add_subplot(222)
    c = fig.add_subplot(223)
    d = fig.add_subplot(224)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(images)
        img, label = sess.run([images, labels])
        a.imshow(img[0])
        a.axis('off')

        b.imshow(img[1])
        b.axis('off')

        c.imshow(img[2])
        c.axis('off')

        d.imshow(img[3])
        d.axis('off')

        plt.show()

        coord.request_stop()
        coord.join(threads)
