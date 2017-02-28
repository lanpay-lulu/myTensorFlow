"""
deep net with batch normalization

"""

import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


N_LAYERS = 3
ACTIVATION = tf.nn.relu
N_HIDDEN_UNITS = 30

def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, acfun=None):
        w = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=0.1))
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(inputs, w) + b
        if acfun is None:
            outputs = wx_plus_b
        else:
            outputs = acfun(wx_plus_b)
        return outputs

    layers_inputs = [xs]

    for l in range(N_LAYERS):
        layer_input = layers_inputs[l]
        in_size = layer_input.get_shape()[1].value
        
        output = add_layer(
            layer_input,    # input
            in_size,        # input size
            N_HIDDEN_UNITS, # output size
            ACTIVATION,     # activation function
        )
        layers_inputs.append(output) 
     
    # output
    py = add_layer(
        layers_inputs[-1], 
        N_HIDDEN_UNITS, 
        10, 
        acfun=None
    )
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py, labels=ys))
    #cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]



def main():
    mnist = input_data.read_data_sets("mnist/input_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    X = tf.placeholder('float', [None, 784])
    Y = tf.placeholder('float', [None, 10])
    
    norm = False
    train_op, cost, layers_inputs = built_net(X, Y, norm)
    
    py_x = layers_inputs[-1] # output
    predict_op = tf.argmax(py_x, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batch_size = 128
        for i in range(100):
            for start in range(0, len(trX), batch_size):
                end = start + batch_size
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
                print(i, np.mean(np.argmax(teY, axis=1) ==
                    sess.run(predict_op, feed_dict={X: teX})))


if __name__ == '__main__':
    main()


