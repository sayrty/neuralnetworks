# read dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# define model (mlp)
import tensorflow as tf

batch_size = 32
#first layer
input = tf.placeholder('float32', [batch_size, 784])

# second layer
#W1 = tf.Variable(tf.random_normal([784, 100]))
#1 = tf.Variable(tf.random_normal([batch_size, 100]))
#hidden = tf.sigmoid(tf.matmul(input, W1) + b1)
hidden=tf.layers.dense(input,100,activation=tf.nn.sigmoid)

# output layer
#W2 = tf.Variable(tf.random_normal([100, 10]))
#b2 = tf.Variable(tf.random_normal([batch_size, 10]))
#output = tf.matmul(hidden, W2) + b2
output=tf.layers.dense(hidden,10,activation=None)

# y
label = tf.placeholder('float32', [batch_size, 10])

# loss
loss = tf.losses.softmax_cross_entropy(label, output)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)

# training operation
train_op = optimizer.minimize(loss)

import time
import numpy as np

### training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(50):
        for i in range(mnist.train.num_examples):
            x, y = mnist.train.next_batch(batch_size)
            cur_loss, _ = sess.run([loss, train_op], feed_dict={input:x, label:y})
            print (cur_loss)
            #time.sleep(0.2)
    print ('training completed ..')

    # for loop ends, training ends
    # now test
    x, y = mnist.test.next_batch(batch_size)
    cur_loss, pred = sess.run([loss, output], feed_dict={input:x, label:y})
    print ('test loss:', cur_loss)
    print (pred[0])
    print (y[0])
    print ('predicted value:', np.argmax(pred[0]))
    print ('actual value:', np.argmax(y[0]))
