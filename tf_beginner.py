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

# OK, return to TF, in TF it out loss-function looks like
# loss = tf.reduce_sum(tf.square(linear_model - y)) / 8
# If you don't understand how, what, why it works, I clear that in codes below.

# We have some data (x,y) |= {(22, 18)，(25, 15)，(28, 12)，(30, 10)......}
# I guss they're coming from a linear-model(Data scientist can give more suggestions.)
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






