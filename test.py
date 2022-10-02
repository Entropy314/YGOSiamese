import tensorflow as tf
import numpy as np

if __name__ == '__main__': 
    for x in range(100):
        with tf.device('/GPU:0'):
            a = tf.Variable(np.random.rand(3000,3000))
            b = tf.Variable(np.random.rand(3000,3000))
            c = tf.matmul(a,b)