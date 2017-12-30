import tensorflow as tf
import numpy as np
import os

from scipy.misc import imread,imresize
from os.path import join
from os import walk


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tfrecords_path', './tfrecords', """path of tfrecords""")
tf.app.flags.DEFINE_string('raw_data_path', './test', """path of raw images""")


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNEL = 3
NUM_CLASS = 2

NUM_EXAPLME_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAPLME_PER_EPOCH_FOR_TEST = 50000

BATCH_SIZE = 32

def read_images(path):
    """Read image from source file/directory

    Args：
        path: source derectory
    Return:
        An object representing all images and labels, fields:
        images: all image data
        labels: all labels
        num: number of images
    """

    # Get a list filenames
    filenames = next(walk(path))[2]
    num_file = len(filenames)

    # Initialize images and labels.
    images = np.zeros((num_file, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype=np.uint8)
    labels = np.zeros((num_file, ), dtype=np.uint8)

    # Iterate/Read all files
    for index, filename in enumerate(filenames):
        # Read single image and resize it to your expected size
        img = imread(join(path, filename))
        img = imresize(img, (IMAGE_HEIGHT, IMAGE_HEIGHT))
        images[index] = img

        # TO DO:
        if filename[0:3] == 'cat':
            labels[index] = int(0)
        else:
            labels[index] = int(1)

        if index % 1000 == 0:
            print("Reading the %sth image" % index)

    class ImgData(object):
        pass

    result = ImgData()
    result.images = images
    result.labels = labels
    result.num = num_file

    return result


def convert(data, destination):
    """Convert images to tfrecords

    Args:
        data: an object of ImgData, consisting of images, labels and number of images
        destination: destination filename of tfrecords
    """

    images = data.images
    labels = data.labels
    num_examples = data.num

    # filenale of tfrecords
    filename = destination

    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image = images[index].tostring()
        label = labels[index]

        # Attention: Example -> Features -> Feature
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename_queue):
    """Read tfrecords file

    Args:
        filename_queue: a list of tfrecords file

    Returns:
        img, label: a single image and label
    """
    filename_queue = tf.train.string_input_producer([filename_queue])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int64)

    return image, label


def distorted_input(filename, batch_size):
    """Construct distorted input for cat_dog training using tfrecords.

    Args:
      filename: Path to the data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # full file name
    filename = './tfrecords/' + filename + '.tfrecords'

    if not os.path.exists(filename):
        print('Transfer images to TF_Records')
        raw_data = read_images(FLAGS.raw_data_path)
        convert(raw_data, filename)
        print('End transfering')

    image, label = read_and_decode(filename)
    print('Filling queue images before starting to train.')
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                    num_threads=16, capacity=3000, min_after_dequeue=1000)

    return images, labels