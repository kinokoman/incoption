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


class Incoption:
	def __init__(self):
		# Data
		if DATA == 'fizzbuzz':
			self.data = DataFizzBuzz().main()
		elif DATA == 'mnist':
			self.data = DataMnist().main()


	def main(self):
		#GA().main()

		print(1)

		# Read best prameters
		df = pd.read_csv(LOG_DIR+LOG_FILE_TOP)
		numbers = df[-1:]['param'].values[0]
		numbers = [int(a) for a in numbers[1:-1].split(', ')]

		print(numbers)
		
		# Train best model and save it
		DeepLearning().main(self.data, numbers)
		
		print(3)

		# Read best model and test it
		DeepLearning().test(self.data, numbers)
		
		"""
		dl = DeepLearning()
		params = Param().convert_param(numbers)     
		model = dl.design_network(self.data, params)
		dl.test_network(self.data, model)
		"""
		

if __name__ == "__main__":
	Incoption().main()

