# coding: utf-8

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
import random
from sklearn.preprocessing import OneHotEncoder
import os
import glob
import time
import copy

from data_fizzbuzz import DataFizzBuzz


class Test:
	def __init__(self):
		pass


	def main(self):
		n_gene = 11
		parent1 = [random.randint(0, 3) for i in range(n_gene)]
		parent2 = [random.randint(0, 3) for i in range(n_gene)]

		sample = random.sample(range(n_gene), n_gene/2)
		sample.sort()

		child = copy.deepcopy(parent1)
		for s in sample:
			child[s] = parent2[s]
	

		print(parent1)
		print(parent2)
		print(sample)
		print(child)



	def save_n_restore_model(self):
		data = DataFizzBuzz().main()
		model = self.design_model(data)
		self.save_model(data, model)
		self.restore_model(data, model)


	def design_model(self, data):
		X  = tf.placeholder(tf.float32, [None, data[0].shape[1]])
		W1 = tf.Variable(tf.truncated_normal([data[0].shape[1], 100], stddev=0.01), name='W1')
		B1 = tf.Variable(tf.zeros([100]), name='B1')
		H1 = tf.nn.tanh(tf.matmul(X, W1) + B1)
		W2 = tf.Variable(tf.random_normal([100, data[1].shape[1]], stddev=0.01), name='W2')
		B2 = tf.Variable(tf.zeros([data[1].shape[1]]), name='B2')
		Y = tf.matmul(H1, W2) + B2
		Y_ = tf.placeholder(tf.float32, [None, data[1].shape[1]])

		tf.add_to_collection('vars', W1)
		tf.add_to_collection('vars', B1)
		tf.add_to_collection('vars', W2)
		tf.add_to_collection('vars', B2)

		model = {'X': X, 'Y': Y, 'Y_': Y_}

		return model


	def save_model(self, data, model):
		# Data
		train_data  = data[0]
		train_label = data[1]
		test_data  = data[2]
		test_label = data[3]
		
		# Model
		X, Y, Y_ = model['X'], model['Y'], model['Y_']

		# Functions
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, Y_))
		step = tf.train.AdamOptimizer(0.05).minimize(loss)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))

		# Setting
		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		# Randamize data
		p = np.random.permutation(range(len(train_data)))
		train_data, train_label = train_data[p], train_label[p]

		# Training
		for start in range(0, train_label.shape[0], 1):
			end = start + 100
			sess.run(step, feed_dict={X: train_data[start:end], Y_: train_label[start:end]})
			
			# Testing
			train_loss = sess.run(loss, feed_dict={X: train_data, Y_: train_label})
			train_accuracy = sess.run(accuracy, feed_dict={X: train_data, Y_: train_label})
			test_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y_: test_label})

		# Accuracy
		std_output = 'Train Loss: %s, \t Train Accuracy: %s, \t Test Accuracy: %s'
		print(std_output % (train_loss, train_accuracy, test_accuracy))
		
		# Save a model
		for f in os.listdir('../model/'):
			os.remove('../model/'+f)
		saver.save(sess, '../model/test_model')
		print('Saved a model.')

		sess.close()

		
	def restore_model(self, data, model):
		# Data
		train_data  = data[0]
		train_label = data[1]
		test_data  = data[2]
		test_label = data[3]
			
		# Model
		X, Y, Y_ = model['X'], model['Y'], model['Y_']

		# Function
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
		
		# Setting
		saver = tf.train.Saver()
		sess = tf.Session()
		#saver = tf.train.import_meta_graph('../model/test_model.meta')
		saver.restore(sess, '../model/test_model')
		print('Restored a model')

		# Testing
		test_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y_: test_label})	
		print('Test Accuracy: %s' % test_accuracy)
		

	def output_10_epochs(self):
		#if epoch % 1000 == 0:
		N_EPOCH = 1000
		for epoch in range(N_EPOCH+1):
			if int(N_EPOCH/10) != 0 and epoch % int(N_EPOCH/10) == 0:
				print('%s/%s' % (epoch, N_EPOCH))
		

	def get_dicts_in_list_from_dataframe(self):
		dict_in_list = [{'x': random.randint(0, 10), 'y': random.randint(1, 5)} for i in range(10)]
		for dil in dict_in_list:
			print(dil)

		df = pd.DataFrame(dict_in_list)
		print(df)

		dict_in_list = df.to_dict('records')
		for dil in dict_in_list:
			print(dil)


	def multisort(self):
		data = [{'x': random.randint(0, 10), 'y': random.randint(1, 5)} for i in range(10)]
		df = pd.DataFrame(data)
		df = df.sort(['x', 'y'], ascending=[False, True])
		df.reset_index(drop=True, inplace=True)

		print(df)


	def onehotencoder(self):
		"""
		onehotencoder
		"""
		X = np.array([random.choice(range(5)) for i in range(10)])
		print(X)

		X = np.array(X).reshape(1, -1)
		print(X)
		
		X = X.transpose()
		print(X)
		
		encoder = OneHotEncoder(n_values=max(X)+1)
		X = encoder.fit_transform(X).toarray()
		print(X)


	def param(self):
		"""
		# FizzBuzz
		X  = tf.placeholder(tf.float32, [None, n_X])
		H1 = self.__make_layer(X, n_X, n_hidden[0], 'random_normal', 'zeros', 'relu')
		Y  = self.__make_layer(H1, n_hidden[0], n_Y, 'random_normal', 'zeros', '')
		Y_ = tf.placeholder(tf.float32, [None, n_Y])
		"""
		param = [2, 1, 0, 0, 0, 1, 2, 4, 1, 2, 2, 1, 0, 1]
		dic = {}

		# 0: Output Weight
		if   param[0] == 0: dic['o_weight'] = 'zeros'
		elif param[0] == 1: dic['o_weight'] = 'ones'
		elif param[0] == 2: dic['o_weight'] = 'random_normal'
		elif param[0] == 3: dic['o_weight'] = 'truncated_normal'

		# 1: Output Standard deviation
		if   param[1] == 0: dic['o_stddev'] = 0.1
		elif param[1] == 1: dic['o_stddev'] = 0.01
		elif param[1] == 2: dic['o_stddev'] = 0.001
		elif param[1] == 3: dic['o_stddev'] = 0.0001
		
		# 2: Output Bias
		if   param[2] == 0: dic['o_bias'] = 'zeros'
		elif param[2] == 1: dic['o_bias'] = 'ones'
		
		# 3: Output Activation Function
		if   param[3] == 0: dic['o_activ'] = ''
		elif param[3] == 1: dic['o_activ'] = 'relu'
		elif param[3] == 2: dic['o_activ'] = 'tanh'
		elif param[3] == 3: dic['o_activ'] = 'softmax'

		# 4: Train Optimaize
		if   param[4] == 0: dic['tr_opt'] = 'GradientDescentOptimizer'
		elif param[4] == 1: dic['tr_opt'] = 'AdamOptimizer'
		
		# 5: Learning Rate
		if   param[5] == 0: dic['tr_rate'] = 0.1
		elif param[5] == 1: dic['tr_rate'] = 0.01
		elif param[5] == 2: dic['tr_rate'] = 0.001
		elif param[5] == 3: dic['tr_rate'] = 0.0001
				
		# 6: Batch Size
		if   param[6] == 0: dic['batch_size'] = 10
		elif param[6] == 1: dic['batch_size'] = 50
		elif param[6] == 2: dic['batch_size'] = 100
		
		# 7: The Number of Iteration
		if   param[7] == 0: dic['n_iter'] = 1
		elif param[7] == 1: dic['n_iter'] = 10
		elif param[7] == 2: dic['n_iter'] = 100
		elif param[7] == 3: dic['n_iter'] = 1000
		elif param[7] == 4: dic['n_iter'] = 10000

		# 8: The Number of hidden layer
		if   param[8] == 0: dic['n_h_layer'] = 0
		elif param[8] == 1: dic['n_h_layer'] = 1
		elif param[8] == 2: dic['n_h_layer'] = 2
		elif param[8] == 3: dic['n_h_layer'] = 3
		
		# 9~: Hidden Layer Design
		idx = 8
		for i in range(dic['n_h_layer']):
			# The Number of Node
			if   param[idx+1+5*i] == 0: dic['h%s_n_node'%(i+1)] = 10
			elif param[idx+1+5*i] == 1: dic['h%s_n_node'%(i+1)] = 50
			elif param[idx+1+5*i] == 2: dic['h%s_n_node'%(i+1)] = 100

			# Output Weight
			if   param[idx+2+5*i] == 0: dic['h%s_n_weight'%(i+1)] = 'zeros'
			elif param[idx+2+5*i] == 1: dic['h%s_n_weight'%(i+1)] = 'ones'
			elif param[idx+2+5*i] == 2: dic['h%s_n_weight'%(i+1)] = 'random_normal'
			elif param[idx+2+5*i] == 3: dic['h%s_n_weight'%(i+1)] = 'truncated_normal'

			# Output Standard deviation
			if   param[idx+3+5*i] == 0: dic['h%s_stddev'%(i+1)] = 0.1
			elif param[idx+3+5*i] == 1: dic['h%s_stddev'%(i+1)] = 0.01
			elif param[idx+3+5*i] == 2: dic['h%s_stddev'%(i+1)] = 0.001
			elif param[idx+3+5*i] == 3: dic['h%s_stddev'%(i+1)] = 0.0001
			
			# Output Bias
			if   param[idx+4+5*i] == 0: dic['h%s_bias'%(i+1)] = 'zeros'
			elif param[idx+4+5*i] == 1: dic['h%s_bias'%(i+1)] = 'ones'
			
			# Output Activation Function
			if   param[idx+5+5*i] == 0: dic['h%s_activ'%(i+1)] = ''
			elif param[idx+5+5*i] == 1: dic['h%s_activ'%(i+1)] = 'relu'
			elif param[idx+5+5*i] == 2: dic['h%s_activ'%(i+1)] = 'tanh'
			elif param[idx+5+5*i] == 3: dic['h%s_activ'%(i+1)] = 'softmax'

		for k in sorted(dic):
			print('%-12s'%k, dic[k])


	def accuracy(self):
		"""
		Prediction Accracy Calculation Test.
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
		"""
		Y = np.array([
						[0.1, 0.2, 0.3, 0.4],
						[0.0, 0.8, 0.2, 0.0],
						[0.0, 0.4, 0.5, 0.1]
					])

		print(Y)

		Y_ = np.array([
						[0.0, 0.0, 1.0, 0.0],
						[0.0, 1.0, 0.0, 0.0],
						[0.0, 0.0, 1.0, 0.0]
					])

		print(Y_)

		sess = tf.Session()
		
		print(sess.run(tf.argmax(Y, 1)))

		print(sess.run(tf.argmax(Y_, 1)))


		eq = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
		print(sess.run(eq))

		print(sess.run(tf.cast(eq, tf.float32)))

		print(sess.run(tf.reduce_mean(tf.cast(eq, tf.float32))))

		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)), tf.float32))
		print(sess.run(accuracy))


		result = np.mean(sess.run(tf.argmax(Y, 1))==sess.run(tf.argmax(Y_, 1)))
		print(sess.run(tf.argmax(Y, 1))==sess.run(tf.argmax(Y_, 1)))
		print(result)



if __name__ == "__main__":
	Test().main()


