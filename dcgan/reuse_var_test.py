import tensorflow as tf
import numpy as np


def get_w(shape, name='sb'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return w

def test1():
    shape = [2, 3]
    with tf.variable_scope("foo"):
        w1 = get_w(shape)
        print('w1 name', w1.name)
        # already exists error 
        w2 = get_w(shape)
        print('w2 name', w2.name)


def test2():
    shape = [2, 3]
    with tf.variable_scope("foo") as scope:
        # scope.reuse_variables() # if use here, will also cause error, cause w has not been initialized.
        w1 = get_w(shape)
        print('w1 name', w1.name)
        scope.reuse_variables() # fine
        # already exists error 
        w2 = get_w(shape)
        print('w2 name', w2.name)

def main():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #test1()
        test2()


if __name__ == '__main__':
    main()
