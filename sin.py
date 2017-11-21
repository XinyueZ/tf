import math

import numpy
import tensorflow as tf

rng = numpy.random

test_name = "Sine"

# Parameters
learning_rate = 0.01
training_epochs = 5000
display_step = 50

# Training Data, target: f(x) = 3 * sin(x) + 5
train_X = numpy.random.normal(0., math.pi, size=360)
train_Y = []
for x in train_X: train_Y.append(3 * math.sin(x) + 5)
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a Sine model: f(x) = W*sin(x) + b
activation = tf.add(tf.multiply(tf.sin(X), W), b)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)  # L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    print "cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), \
        "W=", sess.run(W), "b=", sess.run(b)
