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

BATCH_SIZE = 100 # 
#OPT_RATE = 0.05 # FizzBuzz
OPT_RATE = 0.01 # MNIST for beginner
#N_ITER = 10000 # FizzBuzz
N_ITER = 1 # MNIST for beginner
LOG_PATH = './log/'



class DeepLearning:
    def __init__(self):
        """
        - layer
            - num node
            - weight
            - bias
            - activatoin function

        - loss
            - stddev
        - train step
            - opt rate

        - batch size
        - num iter
        """
        pass


    def main(self):
        print 'Setting data...'
        #data = DataFizzBuzz().main()
        data = DataMnist().main()
        
        print 'Training...'
        model = self.design_model(data)
        self.train_model(data, model)


    def design_model(self, data):
        # Set parameters.
        n_X = data[0].shape[1]
        n_Y = data[1].shape[1]
        #n_hidden = [100, 100]
        #n_hidden = [100]  # FizzBuzz
        n_hidden = [10]  # MNIST for beginner

        # Set the model.
        """
        X  = tf.placeholder(tf.float32, [None, n_X])
        H1 = self.__make_layer(X, n_X, n_hidden[0], 'relu')
        H2 = self.__make_layer(H1, n_hidden[0], n_hidden[1], 'relu')
        Y  = self.__make_layer(H2, n_hidden[1], n_Y, 'softmax')
        Y_ = tf.placeholder(tf.float32, [None, n_Y])
        """

        """
        # FizzBuzz
        X  = tf.placeholder(tf.float32, [None, n_X])
        H1 = self.__make_layer(X, n_X, n_hidden[0], 'random_normal', 'zeros', 'relu')
        Y  = self.__make_layer(H1, n_hidden[0], n_Y, 'random_normal', 'zeros', '')
        Y_ = tf.placeholder(tf.float32, [None, n_Y])
        """

        # MNIST for beginner
        X  = tf.placeholder(tf.float32, [None, n_X])
        Y = self.__make_layer(X, n_X, n_Y, 'zeros', 'zeros', 'softmax')
        Y_ = tf.placeholder(tf.float32, [None, n_Y])
        
        #loss, train_step = self.select_functions(Y, Y_, 0, 0)  # FizzBuzz
        loss, train_step = self.select_functions(Y, Y_, 1, 0)  # MNIST for beginner

        model = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'train_step': train_step}

        return model


    def __make_layer(self, I, n_input, n_output, weight, bias, activ):
        # Weight
        if weight == 'zeros':
            W = tf.Variable(tf.zeros([n_input, n_output]))  # MNIST for beginner
        elif weight == 'ones':
            W = tf.Variable(tf.ones([n_input, n_output]))
        elif weight == 'random_normal':
            W = tf.Variable(tf.random_normal([n_input, n_output], stddev=0.01))  # FizzBuzz
        elif weight == 'truncated_normal':
            W = tf.Variable(tf.truncated_normal([n_input, n_output], stddev=0.01))
        
        # Bias
        if bias == 'zeros':
            B = tf.Variable(tf.zeros([n_output]))  # FizzBuzz, MNIST for beginner
        elif bias == 'ones':
            B = tf.Variable(tf.ones([n_output]))

        # Activation Function
        if activ == '':
            O = tf.matmul(I, W) + B  # FizzBuzz
        elif activ == 'relu':
            O = tf.nn.relu(tf.matmul(I, W) + B)  # FizzBuzz
        elif activ == 'tanh':
            O = tf.nn.tanh(tf.matmul(I, W) + B)
        elif activ == 'softmax':
            O = tf.nn.softmax(tf.matmul(I, W) + B)  # MNIST for beginner

        return O


    def select_functions(self, Y, Y_, loss_type, train_type):
        # Loss
        if loss_type == 0:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))  # FizzBuzz
        elif loss_type == 1:
            loss = -tf.reduce_sum(Y_*tf.log(Y))  # MNIST for beginner

        # Train step
        if train_type == 0:
            train_step = tf.train.GradientDescentOptimizer(OPT_RATE).minimize(loss)  # FizzBuzz, MNIST for beginner
        elif train_type == 1:
            train_step = tf.train.AdamOptimizer(OPT_RATE).minimize(loss)

        return loss, train_step


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

        for epoch in range(N_ITER):  # for FizzBuzz
            p = np.random.permutation(range(len(train_X)))
            train_X, train_Y = train_X[p], train_Y[p]

            # Train
            for start in range(0, train_X.shape[0], BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={X: train_X[start:end], Y_: train_Y[start:end]})

                if start % 100 == 0:
                    lo55 = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
                    accu_train = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                    accu_test = sess.run(accuracy, feed_dict={X: test_X, Y_: test_Y})
                    print start, lo55, accu_train, accu_test
            # Test
            """
            if epoch % 100 == 0:
                lo55 = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
                accu_train = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                accu_test = sess.run(accuracy, feed_dict={X: test_X, Y_: test_Y})
                print epoch, lo55, accu_train, accu_test
            """


if __name__ == "__main__":
    DeepLearning().main()

