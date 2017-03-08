import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib # to plot images
# Force matplotlib to not use any X-server backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("mnist/input_data/")
images = mnist.train.images

def xavier_initializer(shape):
    return tf.random_normal(shape=shape, stddev=1.0/shape[0])

# Generator
z_size = 120
g_w1_size = 300 
g_w2_size = 200
g_out_size = 28 * 28

# Discriminator
x_size = 28 * 28
d_w1_size = 300
d_w2_size = 200
d_out_size = 1

z = tf.placeholder('float', shape=(None, z_size))
X = tf.placeholder('float', shape=(None, x_size))


# use dict to share variables
g_weights = {
    'w1': tf.Variable(xavier_initializer(shape=(z_size, g_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[g_w1_size])),
    'w2': tf.Variable(xavier_initializer(shape=(g_w1_size, g_w2_size))),
    'b2': tf.Variable(tf.zeros(shape=[g_w2_size])),
    'out': tf.Variable(xavier_initializer(shape=(g_w2_size, g_out_size))),
    'bo': tf.Variable(tf.zeros(shape=[g_out_size])),
}

d_weights ={
    'w1': tf.Variable(xavier_initializer(shape=(x_size, d_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[d_w1_size])),
    'w2': tf.Variable(xavier_initializer(shape=(d_w1_size, d_w2_size))),
    'b2': tf.Variable(tf.zeros(shape=[d_w2_size])),
    'out': tf.Variable(xavier_initializer(shape=(d_w2_size, d_out_size))),
    'bo': tf.Variable(tf.zeros(shape=[d_out_size])),
}

def G(z, w=g_weights):
    h1 = tf.tanh(tf.matmul(z, w['w1']) + w['b1'])
    h2 = tf.tanh(tf.matmul(h1, w['w2']) + w['b2'])
    #h1 = tf.nn.relu(tf.matmul(z, w['w1']) + w['b1'])
    return tf.sigmoid(tf.matmul(h2, w['out']) + w['bo'])*255 
    #return tf.sigmoid(tf.matmul(h1, w['out']) + w['b2'])

def D(x, w=d_weights):
    h1 = tf.tanh(tf.matmul(x, w['w1']) + w['b1'])
    h2 = tf.tanh(tf.matmul(h1, w['w2']) + w['b2'])
    #h1 = tf.nn.relu(tf.matmul(x, w['w1']) + w['b1'])
    h3 = tf.matmul(h2, w['out']) + w['bo']
    return h3
    #return tf.sigmoid(h2), h2

def generate_z(n=1):
    return np.random.normal(size=(n, z_size))

sample = G(z)

# Objective functions
dout_real = D(X)
dout_fake = D(G(z))

def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

#G_objective = -tf.reduce_mean(tf.log(delta+D(G(z))))
G_obj = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(dout_fake, tf.ones_like(dout_fake)))
D_obj_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(dout_real, tf.ones_like(dout_real)-0.1)) 
D_obj_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(dout_fake, tf.zeros_like(dout_fake))) 
D_obj = D_obj_real + D_obj_fake

G_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(G_obj, var_list=g_weights.values())
D_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(D_obj, var_list=d_weights.values())

## Training
batch_size = 128
small_batch_size = 32

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(150):
        sess.run(D_opt, feed_dict={
            X: images[np.random.choice(range(len(images)), batch_size)].reshape(batch_size, x_size),
            z: generate_z(batch_size),
        })
        sess.run(G_opt, feed_dict={
            z: generate_z(batch_size)            
        })
        sess.run(G_opt, feed_dict={
            z: generate_z(batch_size)            
        })
        g_cost = sess.run(G_obj, feed_dict={z: generate_z(batch_size)})
        d_cost = sess.run(D_obj, feed_dict={
            X: images[np.random.choice(range(len(images)), batch_size)].reshape(batch_size, x_size),
            z: generate_z(batch_size),
        })
        image = sess.run(G(z), feed_dict={z:generate_z()})
        df = sess.run(tf.sigmoid(dout_fake), feed_dict={z:generate_z()})
        print (i, g_cost, d_cost, image.max(), df[0][0])

    # You may wish to save or plot the image generated
    # to see how it looks like
    for i in range(1):
        image = sess.run(G(z), feed_dict={z:generate_z()})
        image1 = image[0].reshape([28, 28])
        plt.imshow(image1)
        plt.axis('off')
        plt.show()
        #plt.savefig("ppp" + '_vis.png')

        #print image1
        im = Image.fromarray(image1)
        im.show()
