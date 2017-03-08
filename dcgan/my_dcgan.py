from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, image_shape, 
            y_dim=None, z_dim=100, 
            gf_dim = 64, df_dim = 64,
            gfc_dim=1024, dfc_dim=1024):
        self.sess = sess
        self.image_shape = image_shape # [h, w, d]
        self.y_dim = y_dim # label dim
        self.z_dim = z_dim 
        self.gf_dim = gf_dim # generator first conv depth, or last deconv depth
        self.df_dim = df_dim # discriminator first conv depth

        # used when y_dim is not None
        self.gfc_dim = gfc_dim # Dimension of gen units for for fully connected layer 
        self.dfc_dim = dfc_dim # Dimension of discrim units for fully connected layer.

        # other params
        self.c_dim = image_shape[-1] # image depth
        self.batch_size = 64

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        self.inputs = tf.placeholder(
                tf.float32, [self.batch_size] + self.image_shape, name='real_images')
        
        self.z = tf.placeholder(
                tf.float32, [None, self.z_dim], name='z')

        # G used for training, sampler for testing.
        self.G = self.generator(self.z, self.y) 
        self.sampler = self.sampler(self.z, self.y)

        self.D, self.D_logits = ( # D for real
                self.discriminator(self.inputs, self.y, reuse=False)
                )
        self.D_, self.D_logits_ = ( # D for fake
                self.discriminator(self.G, self.y, reuse=True)
                )

        # local function
        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        # define loss
        self.g_loss = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss_real = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
        # summary
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, x, y, config):
        # x shape [?, h, w, d]
        # y shape [?, y_dim]
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        # initialize
        tf.global_variables_initializer().run()
        
        # summary
        self.g_sum = merge_summary(
                [self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
                [self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        # sample vars
        sample_num = self.batch_size
        sample_z = np.random.uniform(-1, 1, size=(sample_num, self.z_dim))
        sample_inputs = x[0:sample_num]
        sample_labels = y[0:sample_num]

        counter = 1
        start_time = time.time()
        
        # start training
        for epoch in range(config.epoch):
            for start in range(0, len(x), self.batch_size):
                batch_idx = int(start / self.batch_size)
                end = start + self.batch_size
                batch_images = x[start:end]
                batch_labels = y[start:end]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                if len(batch_images) < self.batch_size:
                    continue

                # feed and update
                # update D first
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ 
                            self.inputs: batch_images,
                            self.y:batch_labels,
                            self.z: batch_z,
                        })
                self.writer.add_summary(summary_str, counter)

                # update G twice
                for _ in range(1):
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={
                            self.z: batch_z, 
                            self.y:batch_labels,
                        })
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                    self.y:batch_labels
                })
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images,
                    self.y:batch_labels
                })
                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })

                print("Epoch: [%3d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                        % (epoch, batch_idx, time.time() - start_time, errD_fake+errD_real, errG))
                counter += 1

                # sample and save
                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                    )
                    # s(ave 
                    samples = samples
                    save_images(samples, [8, 8],
                            './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, batch_idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


                #if np.mod(counter, 500) == 2:
                #    self.save(config.checkpoint_dir, counter)
                    


    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                # yb is add at the depth dim. for each depth dim, add [0, 0, ..., 1, , ...0]
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)
                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            # layer: z -> h0 -> h1 -> h2 -> h3 -> h4
            # param: s_h16 -> s_h8 -> s_h4 -> s_h2 -> s_h
            if not self.y_dim:
                #s_h, s_w = self.output_height, self.output_width
                s_h, s_w, _ = self.image_shape
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2) 
                self.z_, self.h0_w, self.h0_b = linear(
                        z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                        self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                # value, output shape=[batch, h, w, depth]
                self.h1, self.h1_w, self.h1_b = deconv2d(
                        h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                        h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                        h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                        h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
                return tf.nn.sigmoid(h4)

            else:
                s_h, s_w, _ = self.image_shape
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                        self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                        deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))


    def sampler(self, z, y=None):
        # same as generator, except that for bn layer train=False.
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w, _ = self.image_shape

            if not self.y_dim:
                #s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                h0 = tf.reshape(
                        linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                        [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.sigmoid(h4)
            else:
                #s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))








                                            

