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
    def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None] 'label dim'
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

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

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def sigmoid_cross_entropy_with_logits(x, y):
        try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
    
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
                        s_h, s_w = self.output_height, self.output_width
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
                        return tf.nn.tanh(h4)

                    else:
                        s_h, s_w = self.output_height, self.output_width
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

                        if not self.y_dim:
                            s_h, s_w = self.output_height, self.output_width
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

                            return tf.nn.tanh(h4)
                        else:
                            s_h, s_w = self.output_height, self.output_width
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








                                                

