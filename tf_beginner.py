# coding=utf-8
import tensorflow as tf

# What is loss-function, what is training?
# Human-think:
# loss-function: The smaller the loss, the closer to expectations.
# training: More tries, more closer you are.

# First, I use a simple loss-function named B-P-F-1:
# loss(a,b) = 1/(2 * n) * Σ|activation - input_y|^2
# This loss-function is the only one I think is suitable for beginner,
# we all know it at university.
# For more others my suggestion is to find some special books :)

# If you don't understand how, what, why all works, I clear that in codes below.

# We have some data (x,y) |= {(22, 18)，(25, 15)，(28, 12)，(30, 10)......}
# I guss they're coming from a linear-model(Data-scientist can give more suggestions.)
# Well, our target is the typical model f(x) = a * x + b which we want to train.
# With TF and training we'd find (a, b) in the model which can help us with linear-mode
# to estimate more and more (x, y)s widely from what we've already had.

# Training:
#
# Use loss-function for our model, expectation is loss(a, b) === 0
# (you can decide, near to 0 or absolute equal to 0).
#
# 1. loss(a, b) = 1/(2 * n) * Σx|f(x) - y|^2
# 2. Give random pair (a, b)
# 3. Iterate (x, y) with (a, b)
# 4. Evaluation on loss(a, b)
# 5. When loss(a, b) is limiting to 0,  i.e. 0.0001, if you think it's what you want, stop training
# 6. When loss(a, b) still keep large distance to 0 or your expectation, i.e. 23, go to step 2.

# The whole training with TF is a process to find out data like a, b or c...
# they might call weight, bias in some books, don't be confuse with these names.

# You should define a loss-function.
# You should have some data in hand which will try on loss-function.
# TF updates a, b or c ...
# Generally you should break training with expectation like 0 or 0.0000000000001.
# Sometimes you define times of training, i.e training only 50000 times on your computer.

# Example:
#
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

# Our expected model
activation = tf.add(tf.multiply(a, x), b)  # f(x) = a * x + b.

# Our super-start, the loss-function, my expection is 0, that means:
# the final (a, b) produce a loss(a, b) === 0 with all (x, y)s.
# loss(a,b) = 1/(2 * n) * Σ|activation - input_y|^2
loss = tf.reduce_sum(tf.square(activation - y)) / (2 * 4)  # We have 4 data in hand

# Important: TF can inject x, y into activation and loss automatically.

# Training start

# 1. Ready, it is rule, syntax of TF, don't ask why, it must be done for every training.
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# 2. Inject all data to their positions, and the loss-function will calculated by injections.
# That's the 1st training.
print("loss: %s" % (session.run(loss, {x: x_train, y: y_train})))  # run() do all injections.

# 3. We give a new pair of (a, b).
# That's the 2nd training.
session.run([tf.assign(a, [-1.]),
             tf.assign(b, [40.])])
print("loss: %s" % (session.run(loss, {x: x_train, y: y_train})))  # run() do all injections.

# You see, the 2nd training output is 0.0 !
# Fine!, the target pair of (a, b) is (-1, 40)
