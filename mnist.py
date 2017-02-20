import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype = np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype = np.int32)

max_example = 100
data = data[:max_example]
labels = labels[:max_example]

def display(i):
    img = test_data[i]
    plt.title('Example %d. Label %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28, 28)), cmap = plt.cm.gray_r)

display(0)
