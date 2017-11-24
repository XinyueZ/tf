# coding=utf-8
import tensorflow as tf


# Before using this code-sheet as beginner, please checkout tf_beginner.tf firstly.
# With the tf_beginner.tf you'll understand very basic idea, porch of coding with TF.


# As we all know in the tf_beginner.tf we have enumerated two pairs of (a, b) in order to
# find our the expectation of loss (about 0.0).

# We're so lucky that after only two rounds of training we would get the final (a, b) which
# can make loss(a, b) == 0 with all (x, y)s.

# Should we enumerate randomly every-time?

# The answer is NO.

# What we need is a method which works with loss-function,can enumerate,estimate all future (a, b)s
# automatically. All (a, b)s being output of this method would make loss be near to our expectation.
# This method will update (a, b), every time we call it. The loss will be minimized improved after
# the optimized (a, b) being used in loss-function.
#
# For our example the loss should be minimized near to 0.0, what the expectation is.

# The machine-learning means "learning". Like human, people can learn very "hardly",people can learn
# "so-so" and "progressively". When people learn hardly,they'd know more information, they'd correct
# more what they know. When people learn progressively, they'd correct progressively as well.

# We call this the Learn-Passion, OK, in TF we call this "learning-rate".
# The larger the rate, the greater the difference in assessment results: the loss and (a, b).
# For example:
# we define rate = 0.5, then amplitude of output of loss and (a, b) will be very different
# and unstable after each round of training. On the other hand, if we give a rate = 0.01,
# the difference will be smaller and stable.

# The learning-rate is experience, we can not give an absolute small value or in contrast.

# We call this method: Optimizer
# TF provides a lot build-in Optimizers. For demo we try the GradientDescentOptimizer.

# Same as before, forget what this implements, why, how.
# Firstly we use it, we know that it will update loss and (a, b).
# We make a loop to `re-call` the Optimizer and we'll get the updated loss and (a, b) always.

def format_num(num):
    return "{:.19f}".format(num)


# Example:
#
# This training might not be so hardly, in the contrast it should be
# so-so, step-by-step, progressively.
learning_rate = 0.0028
# We do "training_times" times in this example.
training_times = 400000

# These are data I have from a data-scientist,
# he said it might be from a linear-model like y = a * x + b.
x_train = [22, 25, 28, 30]
y_train = [18, 15, 12, 10]

# Our model will find out the pair (a, b).
# Give first estimation (-1, 50).
a = tf.Variable([-1.], tf.float32)
b = tf.Variable([50.], tf.float32)

# Placeholder for all input-pairs (x, y).
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Our expectation-model
activation = tf.add(tf.multiply(a, x), b)  # f(x) = a * x + b.

# The loss-function, my expectation is 0, that means:
# the final (a, b) produce a loss(a, b) === 0 with all (x, y)s.
# loss(a,b) = 1/(2 * n) * Σ|activation - input_y|^2
loss = tf.reduce_sum(tf.square(activation - y)) / (2 * 4)  # We have 4 data in hand

# Our super-star, the optimizer-function.
# As I say before, we use a build-in one, the GradientDescentOptimizer.
# Don't ask me, why, how, what. Use it firstly.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# We make a loop to build a training-round.
# In tf_beginner.tf it is fixed with "2".
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_times):
        # (a, b) will be updated after being optimized, like in tf_beginner.py, they're variables:
        # session.run([tf.assign(a, [-1.]),
        #              tf.assign(b, [40.])])
        sess.run(optimizer, {x: x_train, y: y_train})
        # Calculate loss with refreshed (a, b) and print them all.
        current_a, current_b, current_loss = sess.run([a, b, loss], {x: x_train, y: y_train})
        print "loss=", format_num(current_loss), "a=", (current_a), "b=", (current_b), ""

    # Training accuracy, as final output of training.
    current_a, current_b, current_loss = sess.run([a, b, loss], {x: x_train, y: y_train})
    print "loss=", format_num(current_loss), "a=", (current_a), "b=", (current_b), ""
