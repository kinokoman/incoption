# coding: utf-8

import random
import math
import copy
import operator
import pandas as pd
import time

from data_fizzbuzz import DataFizzBuzz
from data_mnist import DataMnist
from param import Param
from deeplearning import DeepLearning
from ga import GA
import config

DATA = config.DATA
N_HIDDEN_LAYER = config.N_HIDDEN_LAYER

N_POP = config.N_POP
N_GEN = config.N_GEN
MUTATE_PROB = config.MUTATE_PROB
ELITE_PROB = config.ELITE_PROB

LOG_DIR = config.LOG_DIR
DEBUG = config.DEBUG_GA
LOG_FILE_TOP = config.LOG_FILE_TOP
LOG_FILE_DETAIL = config.LOG_FILE_DETAIL

BEST_PARAM = config.BEST_PARAM


class Incoption:
	def __init__(self):
		# Data
		if DATA == 'fizzbuzz':
			self.data = DataFizzBuzz().main()
		elif DATA == 'mnist':
			self.data = DataMnist().main()


	def main(self):
		GA().main()


	def test(self, mode='test'):
		"""
		Save and restore a model.
		Set mode to 'train' at first and 'test' at second.
		"""
		if mode == 'train':
			# Train a model with best params and save it
			DeepLearning().main(self.data, BEST_PARAM)
		elif mode == 'test':		
			# Restore the trained model and test it
			DeepLearning().test(self.data, BEST_PARAM)
			


if __name__ == "__main__":
	#Incoption().main()
	Incoption().test()

