#coding=utf-8
"""

"""


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import numpy as np

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, b_h, w_h2, b_h2, w_o, b_o, p_keep_hidden):
    # two hidden layer
    h = tf.add(tf.nn.relu(tf.matmul(X, w_h)), b_h)
    h = tf.nn.dropout(h, p_keep_hidden)

    h2 = tf.add(tf.nn.relu(tf.matmul(h, w_h2)), b_h2)
    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.add(tf.matmul(h2, w_o), b_o)


def main():
    mnist = input_data.read_data_sets("mnist/input_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    X = tf.placeholder('float', [None, 784])
    Y = tf.placeholder('float', [None, 10])
    
    hdim1 = 256
    hdim2 = 256

    w_h = init_weight([784, hdim1])
    b_h = init_weight([1, hdim1])
    w_h2 = init_weight([hdim1, hdim2])
    b_h2 = init_weight([1, hdim2])
    w_o = init_weight([hdim2, 10])
    b_o = init_weight([1, 10])

    # histogram 用来记录变量的数值分布图
    # scalar 用来记录变量的数值
    # 记录与展现频率是自己定的，每次运行
    # summary, acc = sess.run([merged, acc_op] 就会进行记录
    tf.summary.histogram('w_h_summ', w_h)
    tf.summary.histogram('w_h2_summ', w_h2)
    tf.summary.histogram('w_o_summ', w_o)

    p_keep_hidden = tf.placeholder("float")
    py_x = model(X, w_h, b_h, w_h2, b_h2, w_o, b_o, p_keep_hidden)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute mean cross entropy (softmax is applied internally)
        train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
        #train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
        tf.summary.scalar('cost', cost)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
        tf.summary.scalar('accuracy', acc_op)

    #predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph) # for 1.0
        merged = tf.summary.merge_all()

        tf.global_variables_initializer().run()
        batch_size = 128
        
        for i in range(100):
            for start in range(0, len(trX), batch_size):
                end = start + batch_size
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_hidden: 0.5})
            summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY,
                                                p_keep_hidden: 0.5})
            writer.add_summary(summary, i)
            print(i, acc)


if __name__ == '__main__':
    main()

