# coding: utf-8

import sys
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf

from data_fizzbuzz import DataFizzBuzz

reload(sys)
sys.setdefaultencoding('utf-8')


NUM_DIGITS = 10
NUM_HIDDEN = 100
BATCH_SIZE = 128 #10
OPT_RATE = 0.0001
LOG_PATH = './log/'



class DeepLearning:
    def __init__(self):
        pass


    def main(self):
        data = DataFizzBuzz().main()
        X, Y, Y_ = self.design_network(data)
        self.train_model(data, X, Y, Y_)


    def train(self):
        df, df_ = Data().main()
        X, Y, Y_ = self.design_network(df_)
        self.train_model(df_, X, Y, Y_)


    def design_network(self, data):
        # Set parameters.
        n_X = data[0].shape[1]
        n_Y = data[1].shape[1]
        #n_hidden = [100, 100]
        n_hidden = [100]

        # Set the model.
        """
        X  = tf.placeholder(tf.float32, [None, n_X])
        H1 = self.__make_layer(X, n_X, n_hidden[0], 'relu')
        H2 = self.__make_layer(H1, n_hidden[0], n_hidden[1], 'relu')
        Y  = self.__make_layer(H2, n_hidden[1], n_Y, 'softmax')
        Y_ = tf.placeholder(tf.float32, [None, n_Y]) 
        """

        X  = tf.placeholder(tf.float32, [None, n_X])
        H1 = self.__make_layer(X, n_X, n_hidden[0], 'relu')
        Y  = self.__make_layer(H1, n_hidden[0], n_Y, '')
        Y_ = tf.placeholder(tf.float32, [None, n_Y])
        
        return X, Y, Y_


    def __make_layer(self, I, n_input, n_output, activ_func):
        W = tf.Variable(tf.random_normal([n_input, n_output], stddev=0.01))
        #W = tf.Variable(tf.zeros([n_input, n_output]))
        #B = tf.Variable(tf.zeros([n_output]))
        #W = tf.Variable(tf.truncated_normal([n_input, n_output], stddev=0.01))
        B = tf.Variable(tf.ones([n_output]))

        if activ_func == 'tanh':
            O = tf.nn.tanh(tf.matmul(I, W) + B)
        elif activ_func == 'relu':
            O = tf.nn.relu(tf.matmul(I, W) + B)
        elif activ_func == 'softmax':
            O = tf.nn.softmax(tf.matmul(I, W) + B)
        else:
            O = tf.matmul(I, W)

        """
        w_h = tf.Variable(tf.random_normal(shape, stddev=0.01))
        h = tf.nn.relu(tf.matmul(X, w_h))
        nw = tf.matmul(h, w_o)
        """

        return O


    def test(self, trX, trY):
        X = tf.placeholder("float", [None, NUM_DIGITS])
        Y = tf.placeholder("float", [None, 4])

        w_h = self.init_weights([NUM_DIGITS, NUM_HIDDEN])
        w_o = self.init_weights([NUM_HIDDEN, 4])

        py_x = self.model(X, w_h, w_o)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

        predict_op = tf.argmax(py_x, 1)

        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for epoch in range(10000):
                p = np.random.permutation(range(len(trX)))
                trX, trY = trX[p], trY[p]

                for start in range(0, len(trX), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

                if epoch % 100 == 0:
                    print start, end
                    print epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY}))

            numbers = np.arange(1, 101)
            teX = np.transpose(self.binary_encode(numbers, NUM_DIGITS))
            teY = sess.run(predict_op, feed_dict={X: teX})
            output = np.vectorize(self.fizz_buzz)(numbers, teY)

            print output



    def train_model(self, data, X, Y, Y_):
        # data
        train_X = data[0]
        train_Y = data[1]
        test_X = data[2]
        test_Y = data[3]
        
        # initailize.
        sess = tf.InteractiveSession()

        # make equations.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        #loss = -tf.reduce_sum(Y_*tf.log(Y))
        #train_step = tf.train.AdamOptimizer(learning_rate=OPT_RATE).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Start to train.
        tf.initialize_all_variables().run()

        for epoch in range(10000):
            for start in range(0, train_X.shape[0], BATCH_SIZE):
                # Train
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={X: train_X[start:end], Y_: train_Y[start:end]})

            # Test
            if epoch % 100 == 0:
                acc = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                print epoch, acc



    def __next_batch(self, df_, index, n):
        if index + n > len(df_.ix[('train')]['data'].values):
            index = 0
        data = df_.ix[('train')]['data'][index:index+n].values
        label = df_.ix[('train')]['label'][index:index+n].values
        index += n

        return data, label


    def profit(self):
        #
        df, df_ = Data().main()
        X, Y, Y_ = self.design_network(df_)
        pred_label = self.predict_label(df_, X, Y)
        df = self.make_prediction_result(df, pred_label)
        self.output_betting_result(df)
        #self.opt_betting()
        

    def predict_label(self, df_, X, Y):
        # initialize
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        # read trained model.
        saver = tf.train.Saver()
        saver.restore(sess, "tennis_model.ckpt")

        # predicted label.
        pred_label = sess.run(Y, feed_dict={X: df_.ix[('test')]['data'].values})

        return pred_label


    def make_prediction_result(self, df, pred_label):
        # make df of prediction label.
        df_pred = pd.DataFrame(pred_label)
        df_pred = df_pred.rename(columns={0: 'my_win_rate', 1: 'opp_win_rate'})
        df_pred[['my_win_pred', 'opp_win_pred']] = df_pred.applymap(lambda x: round(x, 0))
        
        # remake df of test. 
        df_data = df.ix[('test')]['data']
        df_label = df.ix[('test')]['label']
        df = pd.concat([df_data, df_label], axis=1)

        # add prediction label.
        df_pred.index = df.index
        df = pd.concat([df, df_pred], axis=1)

        # set correct.
        df.ix[df['my_win_pred']==df['my_win'], 'correct'] = 0
        df.ix[df['my_win_pred']!=df['my_win'], 'correct'] = 1        
        
        # set profit.
        df.ix[df['correct']==1, 'profit'] = -1.0
        cond1 = (df['correct']==0) & (df['my_win_pred']==1)
        df.ix[cond1, 'profit'] = df.ix[cond1, 'my_odds'] - 1
        cond2 = (df['correct']==0) & (df['opp_win_pred']==1)
        df.ix[cond2, 'profit'] = df.ix[cond2, 'opp_odds'] - 1

        assert len(df_data) == len(df_label) == len(df_pred)

        return df


    def output_betting_result(self, df):
        for i in range(5,10):
            # filter betting target by condition.
            cond_win_rate = (df['my_win_rate']>=i*0.1) | (df['opp_win_rate']>=i*0.1)
            cond_round = df['round'].str.contains('Q-')
            df_ = df.ix[cond_win_rate & ~cond_round]

            # claculate accuracy and profit.
            my_correct = len(df_[df_['my_win_pred']==df_['my_win']])
            my_incorrect = len(df_[df_['my_win_pred']!=df_['my_win']])
            accuracy = my_correct / float(my_correct+my_incorrect)
            profit = df_['profit'].sum()

            # output.
            output_str = 'Min Prediction Rate: %4s \t Correct: %4s \t Incorrect: %4s \t Accuracy: %4s \t Profit: %4s'
            print output_str % (i*0.1, my_correct, my_incorrect, round(accuracy,2), profit)


if __name__ == "__main__":
    DeepLearning().main()

