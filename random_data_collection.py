from numpy.random import RandomState

rdm = RandomState(1)
data_set_size = 128
X = rdm.rand(data_set_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

print X
print Y

print X[0:3]
print Y[0:3]

