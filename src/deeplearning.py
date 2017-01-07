# coding: utf-8

import sys
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
import time
import datetime

from data_fizzbuzz import DataFizzBuzz
from data_mnist import DataMnist
from param import Param

reload(sys)
sys.setdefaultencoding('utf-8')

LOG_PATH = '../log/'


class DeepLearning:
    def __init__(self):
        pass


    def main(self):
        start = time.time()

        print 'Setting data...'
        data = DataFizzBuzz().main()
        #data = DataMnist().main()
        
        print 'Training...'
        seq_nums = Param().generate_random_seq_nums()
        params = Param().convert_param(seq_nums)

        model = self.design_network(data, params)
        self.train_network(data, model, params)

        end = time.time()
        print (end-start)/60, 'minutes trained model.'


    def design_network(self, data, params):
        # Set input/output size.
        n_X = data[0].shape[1]
        n_Y = data[1].shape[1]

        # Make network from params
        if params.has_key('h3_n_node'):
            pass
        elif params.has_key('h2_n_node'):
            pass
        elif params.has_key('h1_n_node'):
            X  = tf.placeholder(tf.float32, [None, n_X])
            H1 = self.__make_layer(X, n_X, params['h1_n_node'], params['h1_weight'], params['h1_stddev'], params['h1_bias'], params['h1_activ'])
            Y  = self.__make_layer(H1, params['h1_n_node'], n_Y, params['o_weight'], params['o_stddev'], params['o_bias'], params['o_activ'])
            Y_ = tf.placeholder(tf.float32, [None, n_Y])
        else:
            pass        

        # Select trainer
        loss, train_step = self.__select_trainer(Y, Y_, params)

        network = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'step': train_step}

        return network


    def __make_layer(self, I, n_input, n_output, weight, std_dev, bias, activ):
        # Weight
        if weight == 'zeros':
            W = tf.Variable(tf.zeros([n_input, n_output]))
        elif weight == 'ones':
            W = tf.Variable(tf.ones([n_input, n_output]))
        elif weight == 'random_normal':
            W = tf.Variable(tf.random_normal([n_input, n_output], stddev=std_dev))
        elif weight == 'truncated_normal':
            W = tf.Variable(tf.truncated_normal([n_input, n_output], stddev=std_dev))
        
        # Bias
        if bias == 'zeros':
            B = tf.Variable(tf.zeros([n_output]))
        elif bias == 'ones':
            B = tf.Variable(tf.ones([n_output]))

        # Activation Function
        if activ == '':
            O = tf.matmul(I, W) + B
        elif activ == 'relu':
            O = tf.nn.relu(tf.matmul(I, W) + B)
        elif activ == 'tanh':
            O = tf.nn.tanh(tf.matmul(I, W) + B)
        elif activ == 'softmax':
            O = tf.nn.softmax(tf.matmul(I, W) + B)

        return O


    def __select_trainer(self, Y, Y_, params):
        if params['tr_opt'] == 'GradientDescentOptimizer':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
            train_step = tf.train.GradientDescentOptimizer(params['tr_rate']).minimize(loss)
        elif params['tr_opt'] == 'AdamOptimizer':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
            train_step = tf.train.AdamOptimizer(params['tr_rate']).minimize(loss)

        return loss, train_step


    def train_network(self, data, network, params):
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
        tf.initialize_all_variables().run()
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
        
        logs = []
        for epoch in range(params['n_iter']+1):
            # Randamize data
            p = np.random.permutation(range(len(train_data)))
            train_data, train_label = train_data[p], train_label[p]

            # Training
            for start in range(0, train_label.shape[0], params['batch_size']):
                end = start + params['batch_size']
                sess.run(step, feed_dict={X: train_data[start:end], Y_: train_label[start:end]})
            
            # Testing
            train_loss = sess.run(loss, feed_dict={X: train_data, Y_: train_label})
            train_accuracy = sess.run(accuracy, feed_dict={X: train_data, Y_: train_label})
            test_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y_: test_label})

            # Logging
            log = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
            logs.append(log)

            if epoch % 100 == 0:
                std_output = 'Epoch: %s, \t Train Loss: %s, \t Train Accuracy: %s, \t Test Accuracy: %s'
                print(std_output % (log['epoch'], log['train_loss'], log['train_accuracy'], log['test_accuracy']))

        # Save logs
        dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame(logs)
        df.to_csv("./log/accuracy_and_error_%s.csv"%dt, index=False)
                

    """
    def design_model(self, data):
        # Set parameters.
        n_X = data[0].shape[1]
        n_Y = data[1].shape[1]
        n_hidden = [50, 50]
        #n_hidden = [100]  # FizzBuzz
        #n_hidden = [10]  # MNIST for beginner

        # Set the model.
        X  = tf.placeholder(tf.float32, [None, n_X])
        H1 = self.__make_layer(X, n_X, n_hidden[0], 'random_normal', 'zeros', 'relu')
        H2 = self.__make_layer(H1, n_hidden[0], n_hidden[1], 'random_normal', 'zeros', 'relu')
        Y  = self.__make_layer(H2, n_hidden[1], n_Y, 'random_normal', 'zeros', '')
        Y_ = tf.placeholder(tf.float32, [None, n_Y])
        
        # FizzBuzz
        X  = tf.placeholder(tf.float32, [None, n_X])
        H1 = self.__make_layer(X, n_X, n_hidden[0], 'random_normal', 'zeros', 'relu')
        Y  = self.__make_layer(H1, n_hidden[0], n_Y, 'random_normal', 'zeros', '')
        Y_ = tf.placeholder(tf.float32, [None, n_Y])
        
        # MNIST for beginner
        X  = tf.placeholder(tf.float32, [None, n_X])
        Y = self.__make_layer(X, n_X, n_Y, 'zeros', 'zeros', '')
        Y_ = tf.placeholder(tf.float32, [None, n_Y])
        
        loss, train_step = self.__select_trainer(Y, Y_, 0)  # FizzBuzz, MNIST for beginner

        model = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'train_step': train_step}

        return model


    def __make_layer(self, I, n_input, n_output, weight, bias, activ):
        # Weight
        if weight == 'zeros':
            W = tf.Variable(tf.zeros([n_input, n_output]))  # MNIST for beginner
        elif weight == 'ones':
            W = tf.Variable(tf.ones([n_input, n_output]))
        elif weight == 'random_normal':
            W = tf.Variable(tf.random_normal([n_input, n_output], stddev=STDDEV))  # FizzBuzz
        elif weight == 'truncated_normal':
            W = tf.Variable(tf.truncated_normal([n_input, n_output], stddev=STDDEV))
        
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


    def __select_trainer(self, Y, Y_, train_type):
        if train_type == 0:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
        elif train_type == 1:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

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

        for epoch in range(N_ITER):  # FizzBuzz
            p = np.random.permutation(range(len(train_X)))
            train_X, train_Y = train_X[p], train_Y[p]

            # Train
            for start in range(0, train_X.shape[0]+1, BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={X: train_X[start:end], Y_: train_Y[start:end]})

                if start % 1000 == 0:
                    lo55 = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
                    accu_train = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                    accu_test = sess.run(accuracy, feed_dict={X: test_X, Y_: test_Y})
                    print start, lo55, accu_train, accu_test

            # Test
            if epoch % 100 == 0:
                lo55 = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
                accu_train = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                accu_test = sess.run(accuracy, feed_dict={X: test_X, Y_: test_Y})
                print epoch, lo55, accu_train, accu_test

        """



if __name__ == "__main__":
    DeepLearning().main()

