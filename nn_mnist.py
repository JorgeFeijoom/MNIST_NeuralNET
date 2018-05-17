import gzip
import cPickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

# Get files with cPickle
f = gzip.open('mnist.pkl.gz', 'rb')

train_set, valid_set, test_set = cPickle.load(f)
f.close()

# Set train data
train_x, train_y = train_set
train_y = one_hot(train_y.astype(int), 10)

# Set validation data
valid_x, valid_y = valid_set
valid_y = one_hot(valid_y.astype(int), 10)

# Set test data
test_x, test_y = test_set
test_y = one_hot(test_y.astype(int), 10)


# ---------------- Visualizing some element of the MNIST dataset --------------


# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print train_y[57]

# -----------------------------------------------------------------------------

# NEURAL NET

# Variable & Placeholders of tensor flow
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.05)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.RMSPropOptimizer(0.001).minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print "---------------------------"
print " Start training process... "
print "---------------------------"

batch_size = 50
error = []

error1 = 15000
error2 = 15000
epoch = 0

while error1 <= error2:
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    epoch += 1

    # Change comparison
    error2 = error1
    error1 = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})

    error.append(error1)

    print "Epoch:", epoch, " Error:", error1
print "-------------------------------------------------------------"


print "---------------------------"
print " Start testing process...  "
print "---------------------------"

result = sess.run(y, feed_dict={x: test_x})
error_count = 0
correct = 0
total = 0

for b, r in zip(test_y, result):
    if np.argmax(b) == np.argmax(r):
        correct += 1
        total += 1
    else:
        error_count += 1
        total += 1
print "Correct elements: ", correct, " | of ", total, " in total.", " |", error_count, " errors."
print "-------------------------------------------------------------"

"""
    0.001 Optimizer   +      50 batch size  =  9328/10000
    0.001 Optimizer   +    1000 batch size  =  9316/10000
    0.005 Optimizer   +    1000 batch size  =  9321/10000
    0.001 Optimizer   +     250 batch size  =  9306/10000
    0.005 Optimizer   +     250 batch size  =  9304/10000
    0.005 Optimizer   +     250 batch size  =  9287/10000
    0.005 Optimizer   +     200 batch size  =  9295/10000
    0.008 Optimizer   +     100 batch size  =  9261/10000
    0.005 Optimizer   +     100 batch size  =  9209/10000
"""
plt.figure(1)
plt.plot(error)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.grid(True)
plt.show()
