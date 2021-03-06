"""

"""


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import numpy as np

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, b_h, w_o, b_o):
    # only one hidden layer
    h = tf.add(tf.nn.sigmoid(tf.matmul(X, w_h)), b_h)
    return tf.add(tf.matmul(h, w_o), b_o)


def main():
    mnist = input_data.read_data_sets("mnist/input_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    X = tf.placeholder('float', [None, 784])
    Y = tf.placeholder('float', [None, 10])

    print trX.shape

    w_h = init_weight([784, 128])
    b_h = init_weight([1, 128])
    w_o = init_weight([128, 10])
    b_o = init_weight([1, 10])


    py_x = model(X, w_h, b_h, w_o, b_o)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute mean cross entropy (softmax is applied internally)
    # is the same with belows:
    # yy = tf.nn.softmax(py_x)
    # cost = -tf.reduce_mean(Y*tf.log(yy))


    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer

    predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batch_size = 128
        
        for i in range(100):
            for start in range(0, len(trX), batch_size):
                end = start + batch_size
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            print(i, np.mean(np.argmax(teY, axis=1) ==
                sess.run(predict_op, feed_dict={X: teX})))
            cc = sess.run(cost, feed_dict={X: teX, Y: teY})
            print('cost:', cc)

if __name__ == '__main__':
    main()

