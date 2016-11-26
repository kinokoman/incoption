# coding: utf-8

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter

from data_fizzbuzz import DataFizzBuzz

reload(sys)
sys.setdefaultencoding("utf-8")


class Test:
    def __init__(self):
        pass


    def main(self):
        Y = np.array([
                        [0.1, 0.2, 0.3, 0.4],
                        [0.0, 0.8, 0.2, 0.0],
                        [0.0, 0.4, 0.5, 0.1]
                    ])

        print Y

        Y_ = np.array([
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]
                    ])

        print Y_

        sess = tf.Session()
        
        print sess.run(tf.argmax(Y, 1))

        print sess.run(tf.argmax(Y_, 1))


        eq = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        print sess.run(eq)

        print sess.run(tf.cast(eq, tf.float32))

        print sess.run(tf.reduce_mean(tf.cast(eq, tf.float32)))

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
        print sess.run(accuracy)


        result = np.mean(sess.run(tf.argmax(Y, 1))==sess.run(tf.argmax(Y_, 1)))
        print sess.run(tf.argmax(Y, 1))==sess.run(tf.argmax(Y_, 1))
        print result



if __name__ == "__main__":
    Test().main()


