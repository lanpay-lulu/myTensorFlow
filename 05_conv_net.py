

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def model(X, w, w2, w3, w4, w_o, p_keep_conv):
    # X size is 4D: [batch, in_height, in_width, in_channels]
    # w is the filter. size is [height, width, in_channels, out_channels]
    # strides = [1, stride, stride, 1] for most case
    # padding "SAME"(zero padding) or "VALID"(no padding). 
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,     # l1a shape=(?, 28, 28, 32)
                        strides=[1,1,1,1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1],   # l1 shape = (?, 14, 14, 32)
                        strides=[1,2,2,1], padding='SAME')
    #l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,     # l2a shape=(?, 14, 14, 64)
                        strides=[1,1,1,1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1],   # l2 shape = (?, 7, 7, 64)
                        strides=[1,2,2,1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,     # l3a shape=(?, 7, 7, 128)
                        strides=[1,1,1,1], padding='SAME'))
    l3 = tf.nn.max_pool(l2a, ksize=[1,2,2,1],   # l3 shape = (?, 4, 4, 128)
                        strides=[1,2,2,1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    py = tf.matmul(l4, w_o)
    return py

def main():
    

    trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
    teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

    X = tf.placeholder("float", [None, 28, 28, 1])
    Y = tf.placeholder("float", [None, 10])

    w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
    w4 = init_weights([128 * 4 * 4, 256]) # FC 128 * 4 * 4 inputs, 625 outputs
    w_o = init_weights([256, 10])         # FC 625 inputs, 10 outputs (labels)

    p_keep_conv = tf.placeholder("float")
    py = model(X, w, w2, w3, w4, w_o, p_keep_conv) # predict y
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py, 1)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batch_size = 128
        test_size = 256
        for i in range(100):
            for start in range(0, len(trX), batch_size):
                end = start + batch_size
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv:0.8})
            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                        sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                        p_keep_conv: 1.0})))










