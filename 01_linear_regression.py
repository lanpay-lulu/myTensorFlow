import tensorflow as tf
import numpy as np


def model(X, w, b):
    return tf.add(tf.matmul(X, w), b)

def main():
    trX = np.linspace(-1, 1, 101)
    trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 + 1.5 # create a y value which is approximately linear but with some random noise

    X = tf.placeholder("float") # create symbolic variables
    Y = tf.placeholder("float")

    w = tf.Variable(0.0, name='w')
    b = tf.Variable(0.0, name='b')
    y_model = model(X, w, b)

    cost = tf.square(Y - y_model)

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

    with tf.Session() as sess:
        # init all variables
        tf.global_variables_initializer().run() 

        for i in range(100):
            for (x, y) in zip(trX, trY):
                sess.run(train_op, feed_dict={X: x, Y: y})

        print(sess.run(w))
        print(sess.run(b))


