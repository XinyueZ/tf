# This example shows how we use exponential_decay of tensorflow
# to generate learning-rate with training-loop.
import numpy
import numpy as np
import tensorflow as tf

rng = numpy.random
N = 3000
OUTPUT_STEP = 5
x_logits = np.random.normal(loc=0.0, scale=1.0, size=(N, 3))
y_logits = 3 * x_logits + 5  # Make some test data

# Linear-mode: f(x) = a * x + b
a = tf.Variable(rng.randn())
b = tf.Variable(rng.randn())

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

activation = a * x + b

# Using exponential_decay to calculate learning-rate dynamically.
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.001, global_step, 200, 0.96, staircase=True)

loss = tf.reduce_sum(tf.square(activation - y)) / (2 * N)  # We have 4 data in hand
optimizer = tf.train.AdamOptimizer(loss).minimize(loss, global_step=global_step)

optimizer_name = "AdamOptimizer"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print optimizer_name

    # Fit all training data
    for step in range(N):
        for (x_, y_) in zip(x_logits, y_logits):
            sess.run(optimizer, feed_dict={x: x_, y: y_})

        # Display logs per epoch step
        if step % OUTPUT_STEP == 0:
            print "Epoch:", '%04d' % (step + 1), "Loss=", \
                "{:.19f}".format(sess.run(loss, feed_dict={x: x_logits, y: y_logits})), \
                "a=", "{:.19f}".format(sess.run(a)), "b=", "{:.19f}".format(sess.run(b))

    print optimizer_name, "Finished!"
    print "Loss=", "{:.19f}".format(sess.run(loss, feed_dict={x: x_logits, y: y_logits})), \
        "a=", "{:.19f}".format(sess.run(a)), "b=", "{:.19f}".format(sess.run(b))
