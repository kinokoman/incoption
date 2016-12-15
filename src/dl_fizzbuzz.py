# coding: utf-8

import sys
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
import time

from data_fizzbuzz import DataFizzBuzz
from data_mnist import DataMnist

reload(sys)
sys.setdefaultencoding('utf-8')


class DLFizzBuzz:
    def __init__(self):
        pass


    def main(self):
        start = time.time()

        # FizzBuzzデータを生成して取得する。
        data = DataFizzBuzz().main()

        # Deep Learnigモデルを設計する。
        model = self.design_model(data)
        
        # Deep Learningモデルを学習させる。
        self.train_model(data, model)

        end = time.time()
        print (end-start)/60, 'minutes trained model.'


    def design_model(self, data):
        # 入力層
        X  = tf.placeholder(tf.float32, [None, data[0].shape[1]])

        # 隠れ層
        W1 = tf.Variable(tf.random_normal([data[0].shape[1], 100], stddev=0.01))
        B1 = tf.Variable(tf.zeros([100]))
        H1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        
        # 出力層
        W2 = tf.Variable(tf.random_normal([100, data[1].shape[1]], stddev=0.01))
        B2 = tf.Variable(tf.zeros([data[1].shape[1]]))
        Y = tf.matmul(H1, W2) + B2

        # 正解
        Y_ = tf.placeholder(tf.float32, [None, data[1].shape[1]])
        
        # 学習関数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        
        model = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'train_step': train_step}

        return model


    def train_model(self, data, model):
        # dataのセット
        train_X = data[0]
        train_Y = data[1]
        test_X = data[2]
        test_Y = data[3]

        # modelのセット
        X = model['X']
        Y = model['Y']
        Y_ = model['Y_']
        loss = model['loss']
        train_step = model['train_step']
        
        # 定義
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))

        # 初期化
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

        for epoch in range(10000):
            # データのランダマイズ
            p = np.random.permutation(range(len(train_X)))
            train_X, train_Y = train_X[p], train_Y[p]

            # 学習
            for start in range(0, train_X.shape[0], 10):
                end = start + 10
                sess.run(train_step, feed_dict={X: train_X[start:end], Y_: train_Y[start:end]})
            
            # テスト
            if epoch % 100 == 0:
                # コスト関数
                lo55 = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
                # 教師データの精度
                accu_train = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                # テストデータの精度
                accu_test = sess.run(accuracy, feed_dict={X: test_X, Y_: test_Y})
                # 標準出力
                print 'Epoch: %s, \t Loss: %-8s, \t Train Accracy: %-8s, \t Test Accracy: %-8s' % (epoch, lo55, accu_train, accu_test)


if __name__ == "__main__":
    DLFizzBuzz().main()

