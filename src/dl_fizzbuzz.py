# coding: utf-8

import sys
import importlib
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
import time

from data_fizzbuzz import DataFizzBuzz
from data_mnist import DataMnist

DROPOUT_RATE = 1.0
TRAIN_RATE = 0.01
N_ITER = 10000
BATCH_SIZE = 10


class DLFizzBuzz:
    def __init__(self):
        pass


    def main(self):
        start = time.time()

        data = DataFizzBuzz().main()
        network = self.design_network(data)
        self.train_network(data, network)

        end = time.time()

        print(round((end-start)/60, 1), 'minutes')


    def design_network(self, data):
        # Input layer
        X  = tf.placeholder(tf.float32, [None, data[0].shape[1]])

        # Hidden layers
        W1 = tf.Variable(tf.random_normal([data[0].shape[1], 100], stddev=0.01))
        B1 = tf.Variable(tf.zeros([100]))
        H1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        H1 = tf.nn.dropout(H1, DROPOUT_RATE)
        
        # Output layer
        W2 = tf.Variable(tf.random_normal([100, data[1].shape[1]], stddev=0.01))
        B2 = tf.Variable(tf.zeros([data[1].shape[1]]))
        Y = tf.matmul(H1, W2) + B2

        # Answer
        Y_ = tf.placeholder(tf.float32, [None, data[1].shape[1]])
        
        # Training function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
        step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        
        network = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'step': step}

        return network


    def train_network(self, data, network):
        # data
        train_data  = data[0]
        train_label = data[1]
        test_data  = data[2]
        test_label = data[3]

        # model
        X  = network['X']
        Y  = network['Y']
        Y_ = network['Y_']
        loss  = network['loss']
        step  = network['step']

        # Setting
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
        
        records = []
        for epoch in range(N_ITER+1):
            # Randamize data
            p = np.random.permutation(range(len(train_data)))
            train_data, train_label = train_data[p], train_label[p]

            # Training
            for start in range(0, train_label.shape[0], BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(step, feed_dict={X: train_data[start:end], Y_: train_label[start:end]})
            
            # Testing
            train_loss = sess.run(loss, feed_dict={X: train_data, Y_: train_label})
            train_accuracy = sess.run(accuracy, feed_dict={X: train_data, Y_: train_label})
            test_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y_: test_label})

            record = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
            records.append(record)

            if epoch % 100 == 0:
                std_output = 'Epoch: %s, \t Train Loss: %s, \t Train Accuracy: %s, \t Test Accuracy: %s'
                print(std_output % (record['epoch'], record['train_loss'], record['train_accuracy'], record['test_accuracy']))

        df = pd.DataFrame(records)
        df.to_csv("./log/accuracy_loss_Q_dropout_E_%s_N_train_E_%s_N_batch_E_%s.csv" % (DROPOUT_RATE, TRAIN_RATE, BATCH_SIZE), index=False)
                

if __name__ == "__main__":
    DLFizzBuzz().main()

