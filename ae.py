#Read Dataset

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)

#Define Model(mlp)

import tensorflow as tf

batch_size=32
#First Layer

input=tf.placeholder('float32',[batch_size,784])

#Second Layer

hidden1=tf.layers.dense(input,256,activation=tf.nn.sigmoid)

#Third Layer

hidden2=tf.layers.dense(hidden1,128,activation=tf.nn.sigmoid)

#Output

output=tf.layers.dense(hidden2,784,activation=None)

#Labels

label=tf.placeholder('float32',[batch_size,784])

#loss

loss=tf.losses.softmax_cross_entropy(label,output)

#optimizer

optimizer=tf.train.GradientDescentOptimizer(0.1)

#Training operation

train_op=optimizer.minimize(loss)

import time
import numpy as np

##Training

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(50):
		for i in range (mnist.train.num_examples):
			x,y=mnist.train.next_batch(batch_size)
			cur_loss,_=sess.run([loss,train_op],feed_dict={input:x,label:x})
			print(cur_loss)
		print("Training completed")
		
		#Testing Datasets
		
		x,y=mnist.test.next_batch(batch_size)
		cur_loss,pred=sess.run([loss,output], feed_dict={input:x,label:x})
		print("test loss:",cur_loss)
		print(pred[0])
		print(y[0])
		print('predicted value:',np.argmax(pred[0]))
		print('actual value:',np.argmax(y[0]))
		
		
