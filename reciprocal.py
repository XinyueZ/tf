import numpy
import tensorflow as tf

rng = numpy.random

test_name = "The reciprocal function"

# Parameters
learning_rate = 0.0001
training_epochs = 5000
display_step = 50

# Training Data, target: f(x) = 1 / x
train_X = numpy.random.normal(1., 100., size=100)
train_Y = []
for x in train_X: train_Y.append((1 / x))
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias_b")

# Construct a Quadratic-formula model: f(x) = 1 / x + bias
activation = W * (1 / X) + b

# Minimize the squared errors
cost = tf.reduce_sum(tf.square(activation - Y)) / 10
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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
                "{:.19f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), \
                "W=", "{:.19f}".format(sess.run(W)), "b=", "{:.19f}".format(sess.run(b))

    print "Optimization Finished!"
    print "cost=", "{:.19f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), \
        "W=", "{:.19f}".format(sess.run(W)), "b=", "{:.19f}".format(sess.run(b))
