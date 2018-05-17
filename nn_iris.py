import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

# 150 samples
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)   # the labels are in the last row. Then we encode them in one hot code

# 70% training
x_data_train = data[:105, 0:4].astype('f4')
y_data_train = one_hot(data[:105, 4].astype(int), 3)

# 15% validation
x_data_val = data[106:128, 0:4].astype('f4')
y_data_val = one_hot(data[106:128, 4].astype(int), 3)

# 15% test
x_data_test = data[129:, 0:4].astype('f4')
y_data_test = one_hot(data[129:, 4].astype(int), 3)

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

time.sleep(3)

print "---------------------------"
print " Start training process... "
print "---------------------------"

batch_size = 20
error = []

for epoch in xrange(100):
    for jj in xrange(len(x_data_train) / batch_size):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error_aux = sess.run(loss, feed_dict={x: x_data_train, y_: y_data_train})
    error.append(error_aux)

    print "==========================================================="
    print "Epoch:", epoch, "Error:", error_aux
    print "==========================================================="

    result = sess.run(y, feed_dict={x: x_data_train})

    for b, r in zip(y_data_train, result):
        print b, "-->", r
    print "------------------------------------------------------------"


print "---------------------------"
print " Start testing process...  "
print "---------------------------"

batch_size = 20

for epoch in xrange(1):
    for jj in xrange(len(x_data_test) / batch_size):
        batch_xs = x_data_test[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_test[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "==========================================================="
    print "Muestra para analizar el aprendizaje: "
    print "Epoch:", epoch, "Error:", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    print "==========================================================="

    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(y_data_test, result):
        print b, "-->", r
    print "------------------------------------------------------------"

plt.figure(1)
plt.plot(error)
plt.xlabel("Error")
plt.ylabel("Iterations")
plt.grid(True)
plt.show()