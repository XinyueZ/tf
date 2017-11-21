import numpy
import tensorflow as tf

rng = numpy.random

test_name = "Quadratic formula"

# Parameters
learning_rate = 0.01
training_epochs = 10000
display_step = 50

# Training Data, target: f(x) = x^2 + 2*x + 1
train_X = numpy.random.normal(1, 100, size=100)
train_Y = pow(train_X, 2) + 2 * train_X + 1
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model

# Set model weights
a = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias_b")
c = tf.Variable(rng.randn(), name="bias_c")

# Construct a linear model: f(x) =a*x^2 + b*x + c
activation = tf.add(tf.add(tf.multiply(a, tf.pow(X, 2)), tf.multiply(b, X)), c)

# Minimize the squared errors
loss_less = 10
loss_more = 1
cost = tf.reduce_sum(tf.where(tf.greater(activation, Y),
                              (activation - Y) * loss_more, (Y - activation) * loss_less))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

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
                "W=", sess.run(a), "b=", sess.run(b), "c=", sess.run(c)

    print "Optimization Finished!"
    print "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), \
        "W=", sess.run(a), "b=", sess.run(b), "c=", sess.run(c)
