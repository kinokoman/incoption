# coding: utf-8

import sys
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf

from data_fizzbuzz import DataFizzBuzz
from data_mnist import DataMnist

reload(sys)
sys.setdefaultencoding('utf-8')


LEARNING_RATE = 0.05
BATCH_SIZE = 10 
N_ITER = 10000
LOG_PATH = '../log/'



class DeepLearning:
    def __init__(self):
        pass


    def main(self):
        start = time.time()

        data = DataFizzBuzz().main()
        model = self.design_model(data)
        self.train_model(data, model)

        end = time.time()
        print (end-start)/60, 'minutes trained model.'


    def design_model(self, data):        
        # Set the model.
        X  = tf.placeholder(tf.float32, [None, data[0].shape[1]])

        W1 = tf.Variable(tf.random_normal([data[0].shape[1], 100], stddev=0.01))
        B1 = tf.Variable(tf.zeros([100]))
        H1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        
        W2 = tf.Variable(tf.random_normal([100, data[1].shape[1]], stddev=0.01))
        B2 = tf.Variable(tf.zeros([data[1].shape[1]]))
        Y = tf.matmul(H1, W2) + B2

        Y_ = tf.placeholder(tf.float32, [None, data[1].shape[1]])
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        
        model = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'train_step': train_step}

        return model


    def train_model(self, data, model):
        # data
        train_X = data[0]
        train_Y = data[1]
        test_X = data[2]
        test_Y = data[3]

        # model
        X = model['X']
        Y = model['Y']
        Y_ = model['Y_']
        loss = model['loss']
        train_step = model['train_step']
        
        # initailize.
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))

        for epoch in range(10000):
            p = np.random.permutation(range(len(train_X)))
            train_X, train_Y = train_X[p], train_Y[p]

            # Train
            for start in range(0, train_X.shape[0], 10):
                end = start + 100
                sess.run(train_step, feed_dict={X: train_X[start:end], Y_: train_Y[start:end]})
            
            # Test
            if epoch % 100 == 0:
                lo55 = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
                accu_train = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                accu_test = sess.run(accuracy, feed_dict={X: test_X, Y_: test_Y})
                print 'Epoch: %s, \t Loss: %-8s, \t Train Accracy: %-8s, \t Test Accracy: %-8s' % (epoch, lo55, accu_train, accu_test)


if __name__ == "__main__":
    DeepLearning().main()

