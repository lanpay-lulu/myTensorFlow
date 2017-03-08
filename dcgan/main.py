import os

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np

from my_dcgan import DCGAN 

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")

FLAGS = flags.FLAGS


def main():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    mnist = input_data.read_data_sets("../mnist/input_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    trX = trX.reshape(-1, 28, 28, 1)
    teX = teX.reshape(-1, 28, 28, 1)

    with tf.Session() as sess:
        
        dcgan = DCGAN(
                sess,
                image_shape = [28, 28, 1],
                y_dim = 10,
                z_dim=100)

        x = np.concatenate((trX, teX), axis=0)
        x = x / 255.
        y = np.concatenate((trY, teY), axis=0)

        dcgan.train(x, y, FLAGS)

if __name__ == '__main__':
    main()


