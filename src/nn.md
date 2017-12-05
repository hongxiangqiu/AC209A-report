---
title: nn.py
---

``` python
# Some code from https://www.tensorflow.org/get_started/mnist/pros

import numpy as np
import pandas as pd

train_df = pd.read_csv('subsample_training.csv')
X_train = train_df.drop(['business_id','user_id'],axis=1).as_matrix()
test_df = pd.read_csv('subsample_test.csv')
X_test = test_df.drop(['business_id','user_id'],axis=1).as_matrix()

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

input_ = tf.placeholder(tf.float32, [None, 79])
shuffled_input = tf.random_shuffle(input_)

# correct star
y_ = tf.reshape(shuffled_input[:,0], [-1,1])

# input data
x = tf.reshape(shuffled_input[:,1:], [-1,78])

# build the network
keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
x_drop = tf.nn.dropout(x, keep_prob=keep_prob_input)

W_fc1 = weight_variable([78, 500])
b_fc1 = bias_variable([500])

h_fc1 = tf.nn.relu(tf.matmul(x_drop, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([500, 10])
b_fc2 = bias_variable([10])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([10, 1])
b_fc3 = bias_variable([1])

# range is between 1.0 and 5.0
y = tf.add(tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)*4, 1.0, name="y")
# define the loss function
loss = tf.reduce_mean(tf.square(y - y_))

# define training step and accuracy
learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

y_mean = tf.reduce_mean(y_)
total_error = tf.reduce_sum(tf.square(y_ - y_mean))
explained_error = tf.reduce_sum(tf.square(y_ - y))
r2 = tf.subtract(1.0, tf.div(explained_error, total_error), name='r2')

# create a saver
saver = tf.train.Saver()

# initialize the graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# train
init_rate = 0.0001
print("Start training...")
saver.save(sess, './nn.model')
saver.restore(sess, './nn.model')
order = np.arange(len(X_train))
for i in range(7600):

    sess.run(train_step, feed_dict={input_:X_train, keep_prob_input: 0.8, keep_prob: .5,
                                    learning_rate:init_rate})
    if i%600 == 599:
        init_rate /= 2.0
        # DO NOT BE TOO SMALL
        init_rate = max(init_rate,5e-6)
    if i%300==0:
        print('Training loss:',sess.run(loss, feed_dict={
            x: X_train[:,1:], y_: X_train[:,0].reshape(-1,1), keep_prob_input: 1.0, keep_prob: 1.0}))
        trainr2 = sess.run(r2, feed_dict={
            x: X_train[:,1:], y_: X_train[:,0].reshape(-1,1), keep_prob_input: 1.0, keep_prob: 1.0})
        print("training r2: %f" % trainr2)
        saver.save(sess, './nn.model')
    
testr2 = sess.run(r2, feed_dict={
            x: X_test[:,1:], y_: X_test[:,0].reshape(-1,1), keep_prob_input: 1.0, keep_prob: 1.0})
print("test r2: %f" % testr2)
saver.save(sess, './nn.model')
```
