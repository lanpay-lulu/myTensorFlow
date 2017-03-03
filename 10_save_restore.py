"""

"""


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import os

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

    # save net
    ckpt_dir = "./ckpt_dir"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Call this after declaring all tf.Variables.
    saver = tf.train.Saver()
    
    # This variable won't be stored, since it is declared after tf.train.Saver()
    non_storable_variable = tf.Variable(777)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batch_size = 128
        
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

        start = global_step.eval() # get last global_step
        print("Start from:", start)
        
        for i in range(start, 100):
            for start in range(0, len(trX), batch_size):
                end = start + batch_size
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            global_step.assign(i).eval() # set and update(eval) global_step with index, i
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
            print(i, np.mean(np.argmax(teY, axis=1) ==
                sess.run(predict_op, feed_dict={X: teX})))



if __name__ == '__main__':
    main()

