

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


input_vec_size = lstm_size = 28
time_step_size = 28


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def model(X, w, b, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    # X is a1,a2  b1,b2  c1,c2
    #      a3,a4  b3,b4  c3,c4
    xt = tf.transpose(X, [1,0,2]) # exchange batch_size and time_step_size
    # xt shape: (time_step_size, batch_size, input_vec_size)
    xr = tf.reshape(xt, [-1, lstm_size])
    # XR shape: (time_step_size * batch_size, input_vec_size)
    x_split = tf.split(0, time_step_size, xr) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)
    # x_split is a1,a2  a3,a4
    #            b1,b2  b3,b4
    #            c1,c2  c3,c4

    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    
    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    #inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size]
    outputs, _states = tf.nn.rnn(lstm, x_split, dtype=tf.float32)
    
    # Linear activation, for softmax
    return tf.matmul(outputs[-1], w) + b, lstm.state_size # State size to initialize the stat
    


def main():
    mnist = input_data.read_data_sets("mnist/input_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    trX = trX.reshape(-1, 28, 28)  # 28x28 input img
    teX = teX.reshape(-1, 28, 28)  # 28x28 input img

    X = tf.placeholder("float", [None, 28, 28])
    Y = tf.placeholder("float", [None, 10])

    w = init_weights([lstm_size, 10])
    b = init_weights([10])

    py, state_size = model(X, w, b, lstm_size) # predict y
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py, 1)
    
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batch_size = 128
        test_size = 256
        for i in range(100):
            for start in range(0, len(trX), batch_size):
                end = start + batch_size
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                        sess.run(predict_op, feed_dict={X: teX[test_indices]})))



if __name__ == '__main__':
    main()




