#Read Dataset

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)

#Define Model(mlp)

import tensorflow as tf

batch_size=32

#First Layer

input=tf.placeholder('float32',[batch_size,784])

#Reshape it to 4D Model

reshaped_input=tf.reshape(input,[batch_size,28,28,1])

#Feature Extraction

#First Feature Extraction
##First Convolution

conv1=tf.layers.conv2d(reshaped_input,16,3,activation=tf.nn.relu)

##First Pooling

conv1=tf.layers.max_pooling2d(conv1,2,2)

#Second Feature Extraction 
##Second Convolution

conv2=tf.layers.conv2d(conv1,32,3,activation=tf.nn.relu)

##Second Pooling

conv2=tf.layers.max_pooling2d(conv2,2,2)

#Flatten it from 4D to 2D to form a fully connected network

fc1=tf.contrib.layers.flatten(conv2)

#Classifaction
#Fully connected network

fc1=tf.layers.dense(fc1,512)

#softmax function to classify

softmax_input=tf.layers.dense(fc1,10)

#output

output=tf.nn.softmax(softmax_input)

label = tf.placeholder('float32', [batch_size, 10])

# loss
loss = tf.losses.softmax_cross_entropy(label, output)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)

# training operation
train_op = optimizer.minimize(loss)


#Run(copy)

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
