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
        model_train = self.design_model(data, mode='TRAIN')
        model_test = self.design_model(data, mode='TEST')
        
        # Deep Learningモデルを学習させる。
        self.train_model(data, model_train, model_test)

        end = time.time()
        print (end-start)/60, 'minutes trained model.'


    def design_model(self, data, mode='TRAIN'):
        # 入力層
        X  = tf.placeholder(tf.float32, [None, data[0].shape[1]])

        # 隠れ層
        W1 = tf.Variable(tf.random_normal([data[0].shape[1], 100], stddev=0.01))
        B1 = tf.Variable(tf.zeros([100]))
        H1 = tf.nn.relu(tf.matmul(X, W1) + B1)
        if mode == 'TRAIN':
            H1 = tf.nn.dropout(H1, 0.50)
        
        # 出力層
        W2 = tf.Variable(tf.random_normal([100, data[1].shape[1]], stddev=0.01))
        B2 = tf.Variable(tf.zeros([data[1].shape[1]]))
        Y = tf.matmul(H1, W2) + B2

        # 正解
        Y_ = tf.placeholder(tf.float32, [None, data[1].shape[1]])
        
        # 学習関数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # 精度
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
        
        model = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'train_step': train_step, 'accuracy': accuracy}

        return model


    def train_model(self, data, model_train, model_test):
        # dataのセット
        train_X = data[0]
        train_Y = data[1]
        test_X = data[2]
        test_Y = data[3]

        # modelのセット
        trX = model_train['X']
        trY = model_train['Y']
        trY_ = model_train['Y_']
        trLoss = model_train['loss']
        trStep = model_train['train_step']
        trAccu = model_train['accuracy']
        
        teX = model_test['X']
        teY = model_test['Y']
        teY_ = model_test['Y_']
        teLoss = model_test['loss']
        teStep = model_test['train_step']
        teAccu = model_test['accuracy']
        
        # 初期化
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()

        # Logging data for TensorBoard
        _ = tf.scalar_summary('train loss', trLoss)
        _ = tf.scalar_summary('train accuracy', trAccu)
        _ = tf.scalar_summary('test loss', teLoss)
        _ = tf.scalar_summary('test accuracy', teAccu)
        writer = tf.train.SummaryWriter('./log/', graph_def=sess.graph)

        for epoch in range(10000+1):
            # データのランダマイズ
            p = np.random.permutation(range(len(train_X)))
            train_X, train_Y = train_X[p], train_Y[p]

            # 学習
            for start in range(0, train_X.shape[0], 10):
                end = start + 10
                sess.run(trStep, feed_dict={trX: train_X[start:end], trY_: train_Y[start:end]})
            
            # テスト
            if epoch % 100 == 0:
                # 教師データのコスト関数
                loss_train = sess.run(trLoss, feed_dict={trX: train_X, trY_: train_Y})
                # 教師データの精度
                accu_train = sess.run(trAccu, feed_dict={trX: train_X, trY_: train_Y})
                # テストデータのコスト関数
                loss_test = sess.run(teLoss, feed_dict={teX: test_X, teY_: test_Y})
                # テストデータの精度
                accu_test = sess.run(teAccu, feed_dict={teX: test_X, teY_: test_Y})
                # 標準出力
                std_output = 'Epoch: %s, \t Train Loss: %s, \t Train Accracy: %s, \t Test Loss: %s, \t Test Accracy: %s'
                print std_output % (epoch, round(loss_train, 6), round(accu_train, 6), round(loss_test, 6), round(accu_test, 6))
                
            # Write log to TensorBoard
            summary_str = sess.run(tf.merge_all_summaries(), feed_dict={trX: train_X, trY_: train_Y, teX: test_X, teY_: test_Y})
            writer.add_summary(summary_str, epoch)


        def train_model_2(self, data, model):
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

            # Logging data for TensorBoard
            _ = tf.scalar_summary('loss', loss)
            _ = tf.scalar_summary('accuracy', accuracy)
            writer = tf.train.SummaryWriter('./log/', graph_def=sess.graph_def)

            for epoch in range(10000+1):
                # データのランダマイズ
                p = np.random.permutation(range(len(train_X)))
                train_X, train_Y = train_X[p], train_Y[p]

                # 学習
                for start in range(0, train_X.shape[0], 10):
                    end = start + 10
                    sess.run(train_step, feed_dict={X: train_X[start:end], Y_: train_Y[start:end]})
                
                # テスト
                if epoch % 100 == 0:
                    # 教師データのコスト関数
                    loss_train = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
                    # 教師データの精度
                    accu_train = sess.run(accuracy, feed_dict={X: train_X, Y_: train_Y})
                    # テストデータのコスト関数
                    loss_test = sess.run(loss, feed_dict={X: test_X, Y_: test_Y})
                    # テストデータの精度
                    accu_test = sess.run(accuracy, feed_dict={X: test_X, Y_: test_Y})
                    # 標準出力
                    std_output = 'Epoch: %s, \t Train Loss: %s, \t Train Accracy: %s, \t Test Loss: %s, \t Test Accracy: %s'
                    print std_output % (epoch, round(loss_train, 6), round(accu_train, 6), round(loss_test, 6), round(accu_test, 6))
                    
                # Write log to TensorBoard
                summary_str = sess.run(tf.merge_all_summaries(), feed_dict={X: test_X, Y_: test_Y})
                writer.add_summary(summary_str, epoch)



if __name__ == "__main__":
    DLFizzBuzz().main()

