"""
deep net with batch normalization

"""

import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


N_LAYERS = 2
ACTIVATION = tf.nn.relu
N_HIDDEN_UNITS = 128

def built_net(xs, ys, bn):
    def add_layer(inputs, in_size, out_size, acfun=None):
        w = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=0.1))
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(inputs, w) + b

        if bn:
            fc_mean, fc_var = tf.nn.moments(
                    wx_plus_b,
                    axes=[0], # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001
            
            # apply moving average for mean and var when train on batch
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)
            mean, var = mean_var_with_update()

            wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean, var, shift, scale, epsilon)
            # similar with this two steps:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

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
    layers_inputs.append(py)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py, labels=ys))
    #cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]



def main():
    mnist = input_data.read_data_sets("mnist/input_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    X = tf.placeholder('float', [None, 784])
    Y = tf.placeholder('float', [None, 10])
    
    bn = True
    train_op, cost, layers_inputs = built_net(X, Y, bn)
    
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


