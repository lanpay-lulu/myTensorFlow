#coding:utf-8
# Inspired by https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html

"""
这里word2vec模型选择包括：
    cbow model. 即y是w(t)，x是(w(t-1)，w(t+1))
    skip gram model. 即用w(t)预测w(t-1), w(t+1)    

    负例采样。即当输出one-hot时，由于维度太大，会很慢，此时可以只选择计算其中一批采样维度的loss贡献值。
"""


import collections
import numpy as np
import tensorflow as tf

sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]

batch_size = 20
embedding_size = 2
num_sampled = 15    # Number of negative examples to sample.


def main():
    words = " ".join(sentences).split()
    count = collections.Counter(words).most_common()
    print ("Word count", count[:5])
    # Build dictionaries
    rdic = [i[0] for i in count] #reverse dic, idx -> word
    dic = {w: i for i, w in enumerate(rdic)} #dic, word -> id
    voc_size = len(dic)

    # Make indexed word data
    data = [dic[word] for word in words]
    print('Sample data', data[:10], [rdic[t] for t in data[:10]])

    # Let's make a training data for window size 1 for simplicity
    # ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
    cbow_pairs = [];
    for i in range(1, len(data)-1) :
        cbow_pairs.append([[data[i-1], data[i+1]], data[i]]);
    print('Context pairs', cbow_pairs[:10])

    # Let's make skip-gram pairs
    # (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
    skip_gram_pairs = [];
    for c in cbow_pairs:
        skip_gram_pairs.append([c[1], c[0][0]])
        skip_gram_pairs.append([c[1], c[0][1]])
    print('skip-gram pairs', skip_gram_pairs[:5])
    
    def generate_batch(size):
        assert size < len(skip_gram_pairs)
        x_data=[]
        y_data = []
        r = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)
        for i in r:
            x_data.append(skip_gram_pairs[i][0])  # n dim
            y_data.append([skip_gram_pairs[i][1]])  # n, 1 dim
        return x_data, y_data

    print ('Batches (x, y)', generate_batch(3))


    # Input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    # need to shape [batch_size, 1] for nn.nce_loss
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
   
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
                tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs) # lookup table

    nce_weights = tf.Variable(
                tf.random_uniform([voc_size, embedding_size],-1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([voc_size]))

    # It automatically draws negative samples when we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled, voc_size))

    train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for step in range(100):
            batch_inputs, batch_labels = generate_batch(batch_size)
            _, loss_val = sess.run([train_op, loss],
                    feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
            if step % 10 == 0:
                print("Loss at ", step, loss_val) # Report the loss
        # Final embeddings are ready for you to use. Need to normalize for practical use
        trained_embeddings = embeddings.eval()

    

if __name__ == '__main__':
    main()

