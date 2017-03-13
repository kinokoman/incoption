# coding: utf-8

import sys
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
import time
import datetime
import os

from data_fizzbuzz import DataFizzBuzz
from data_mnist import DataMnist
from param import Param
import config

DEBUG = config.DEBUG_DL
MODEL_DIR = config.MODEL_DIR
MODEL_NAME = config.MODEL_NAME
LOG_DIR = config.LOG_DIR
LOG_FILE_TRAIN = config.LOG_FILE_TRAIN


class DeepLearning:
	def __init__(self):
		pass


	def main(self, data, numbers):
		start = time.time()
		
		params = Param().convert_param(numbers)
		model = self.design_model(data, params)
		log = self.train_model(data, model, params)

		end = time.time()
		time_cost = (end-start)/60

		return log['test_accuracy'], time_cost


	def design_model(self, data, params):
		# Set input/output size.
		n_X = data[0].shape[1]
		n_Y = data[1].shape[1]

		# Make model from params
		if 'h3_n_node' in params:
			pass
		elif 'h2_n_node' in params:
			X  = tf.placeholder(tf.float32, [None, n_X])
			H1 = self.__make_layer(X, n_X, params['h1_n_node'], params['h1_weight'], params['h1_stddev'], params['h1_bias'], params['h1_activ'], '1')
			H2 = self.__make_layer(H1, params['h1_n_node'], params['h2_n_node'], params['h2_weight'], params['h2_stddev'], params['h2_bias'], params['h2_activ'], '2')
			Y  = self.__make_layer(H2, params['h2_n_node'], n_Y, params['o_weight'], params['o_stddev'], params['o_bias'], params['o_activ'], '0')
			Y_ = tf.placeholder(tf.float32, [None, n_Y])
		elif 'h1_n_node' in params:
			X  = tf.placeholder(tf.float32, [None, n_X])
			H1 = self.__make_layer(X, n_X, params['h1_n_node'], params['h1_weight'], params['h1_stddev'], params['h1_bias'], params['h1_activ'], '1')
			Y  = self.__make_layer(H1, params['h1_n_node'], n_Y, params['o_weight'], params['o_stddev'], params['o_bias'], params['o_activ'], '0')
			Y_ = tf.placeholder(tf.float32, [None, n_Y])
		else:
			X  = tf.placeholder(tf.float32, [None, n_X])
			Y  = self.__make_layer(X, n_X, n_Y, params['o_weight'], params['o_stddev'], params['o_bias'], params['o_activ'], '0')
			Y_ = tf.placeholder(tf.float32, [None, n_Y])

		# Select trainer
		loss, train_step = self.__select_trainer(Y, Y_, params)

		model = {'X': X, 'Y': Y, 'Y_': Y_, 'loss': loss, 'step': train_step}

		return model


	def __make_layer(self, I, n_input, n_output, weight, std_dev, bias, activ, layer_no):
		# Weight
		if weight == 'zeros':
			W = tf.Variable(tf.zeros([n_input, n_output]), name='W'+layer_no)
		elif weight == 'ones':
			W = tf.Variable(tf.ones([n_input, n_output]), name='W'+layer_no)
		elif weight == 'random_normal':
			W = tf.Variable(tf.random_normal([n_input, n_output], stddev=std_dev), name='W'+layer_no)
		elif weight == 'truncated_normal':
			W = tf.Variable(tf.truncated_normal([n_input, n_output], stddev=std_dev), name='W'+layer_no)
		
		# Bias
		if bias == 'zeros':
			B = tf.Variable(tf.zeros([n_output]), name='B'+layer_no)
		elif bias == 'ones':
			B = tf.Variable(tf.ones([n_output]), name='B'+layer_no)

		# Activation Function
		if activ == '':
			O = tf.matmul(I, W) + B
		elif activ == 'relu':
			O = tf.nn.relu(tf.matmul(I, W) + B)
		elif activ == 'tanh':
			O = tf.nn.tanh(tf.matmul(I, W) + B)
		elif activ == 'softmax':
			O = tf.nn.softmax(tf.matmul(I, W) + B)

		tf.add_to_collection('vars', 'W'+layer_no)
		tf.add_to_collection('vars', 'B'+layer_no)

		return O


	def __select_trainer(self, Y, Y_, params):
		if params['tr_opt'] == 'GradientDescentOptimizer':
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
			train_step = tf.train.GradientDescentOptimizer(params['tr_rate']).minimize(loss)
		elif params['tr_opt'] == 'AdamOptimizer':
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
			train_step = tf.train.AdamOptimizer(params['tr_rate']).minimize(loss)

		return loss, train_step


	def train_model(self, data, model, params, save_model=False, log_train=False):
		# Data
		train_data  = data[0]
		train_label = data[1]
		test_data  = data[2]
		test_label = data[3]

		# Model
		X  = model['X']
		Y  = model['Y']
		Y_ = model['Y_']
		loss  = model['loss']
		step  = model['step']

		# Setting
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
		
		logs = []
		n_epoch = params['n_epoch']
		for epoch in range(1, n_epoch+1):
			# Randamize data
			p = np.random.permutation(range(len(train_data)))
			train_data, train_label = train_data[p], train_label[p]

			# Training
			for start in range(0, train_label.shape[0], params['n_batch']):
				end = start + params['n_batch']
				sess.run(step, feed_dict={X: train_data[start:end], Y_: train_label[start:end]})
			
			# Testing
			train_loss = sess.run(loss, feed_dict={X: train_data, Y_: train_label})
			train_accuracy = sess.run(accuracy, feed_dict={X: train_data, Y_: train_label})
			test_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y_: test_label})

			# Logging
			log = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
			logs.append(log)

			if DEBUG == True:
				if (int(n_epoch/10) != 0 and epoch % int(n_epoch/10) == 0) or int(n_epoch/10) == 0:
					std_output = 'Epoch: %s, \t Train Loss: %s, \t Train Accuracy: %s, \t Test Accuracy: %s'
					print(std_output % (log['epoch'], log['train_loss'], log['train_accuracy'], log['test_accuracy']))

		# Save trained model
		if save_model == True:
			for f in os.listdir(MODEL_DIR):
				os.remove(MODEL_DIR+f)
			saver = tf.train.Saver()
			saver.save(sess, MODEL_DIR+MODEL_NAME)

		# Save logs
		if log_train == True:
			df = pd.DataFrame(logs)
			df.to_csv(LOG_DIR+LOG_FILE_TRAIN, index=False)
			
		return logs[-1]


	"""
	TRAIN
	"""
	def train(self, data, numbers):
		params = Param().convert_param(numbers)
		model = self.design_model(data, params)
		log = self.train_model(data, model, params, save_model=True, log_train=True)


	"""
	TEST
	"""
	def test(self, data, numbers):
		params = Param().convert_param(numbers)     
		model = self.design_model(data, params)
		self.test_model(data, model)

		print('')
		print('best params  : %s' % str(numbers))
		print('')
		for k,v in sorted(params.items()):
			print('%-13s: %s' % (k, v))


	def test_model(self, data, model):
		# Data
		train_data  = data[0]
		train_label = data[1]
		test_data  = data[2]
		test_label = data[3]

		# Model
		X  = model['X']
		Y  = model['Y']
		Y_ = model['Y_']

		# Setting
		sess = tf.InteractiveSession()
		saver = tf.train.Saver()
		saver.restore(sess, MODEL_DIR+MODEL_NAME)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
		
		# Testing
		test_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y_: test_label})
		print('Test Accuracy: %s' % test_accuracy)
		


if __name__ == "__main__":
	DeepLearning().main()

