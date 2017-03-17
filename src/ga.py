# coding: utf-8
import random
import math
import copy
import operator
import pandas as pd
import time

from data_fizzbuzz import DataFizzBuzz
from data_mnist import DataMnist
from data_cifar10 import DataCifar10
from param import Param
from deeplearning import DeepLearning
import config

DATA = config.DATA
N_HIDDEN_LAYER = config.N_HIDDEN_LAYER

N_POP = config.N_POP
N_GEN = config.N_GEN
MUTATE_PROB = config.MUTATE_PROB
ELITE_PROB = config.ELITE_PROB

CROSSOVER_TYPE = config.CROSSOVER_TYPE

LOG_DIR = config.LOG_DIR
DEBUG = config.DEBUG_GA
LOG_FILE_TOP = config.LOG_FILE_TOP
LOG_FILE_DETAIL = config.LOG_FILE_DETAIL


class GA:
	def __init__(self):
		# Data
		if DATA == 'fizzbuzz':
			self.data = DataFizzBuzz().main()
		elif DATA == 'mnist':
			self.data = DataMnist().main()
		elif DATA == 'cifar10':
			self.data = DataCifar10().main()

		# Parameters
		self.param_ranges = Param().get_param_ranges(N_HIDDEN_LAYER)
		
		# Logs
		self.fitness_master = {}
		self.log_top = []
		self.log_detail = []


	def main(self):
		start = time.time()

		# 1st generation
		pop = [{'param': p} for p in self.get_population()]
		fitness = self.evaluate(pop)

		# Debug
		if DEBUG == True:
			self.debug(1, fitness)

		# After 2nd generation
		for g in range(N_GEN-1):
			# Get elites
			elites = fitness[:int(N_POP*ELITE_PROB)]

			# Crossover and mutation
			pop = elites[:]
			while len(pop) < N_POP:
				if random.random() < MUTATE_PROB:
					m = random.randint(0, len(elites)-1)
					child = self.mutate(elites[m]['param'])
				else:
					c1 = random.randint(0, len(elites)-1)
					c2 = random.randint(0, len(elites)-1)
					child = self.crossover(elites[c1]['param'], elites[c2]['param'])

				# Add a child that was not existed former generations.
				df_pop = pd.DataFrame(pop)
				df_pop['param'] = df_pop['param'].astype(str)
				if len(df_pop.ix[df_pop['param']==str(child)]) == 0:
					pop.append({'param': child})

			# Evaluation
			fitness = self.evaluate(pop)

			# Debug
			if DEBUG == True:
				self.debug(g+2, fitness)

		end = time.time()
		print('Took %s minutes.' % str(int((end-start)/60)))


	def get_population(self):
		# Make population
		pop = []
		for i in range(N_POP):
			ind = Param().make_param(N_HIDDEN_LAYER)
			pop.append(ind)

		return pop


	def clac_score(self, indivisual):
		# Temp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		#"""
		self.log_detail.append({'param': indivisual})
		df = pd.DataFrame(self.log_detail)
		df.to_csv(LOG_DIR+LOG_FILE_DETAIL, index=False)
		#"""
		
		test_accuracy, time_cost = DeepLearning().main(self.data, indivisual)

		dic = {}
		dic['score0'] = test_accuracy
		dic['score1'] = time_cost

		return dic


	def evaluate(self, pop):
		fitness = []
		for p in pop:
			if not 'score0' in p:
				if str(p['param']) in self.fitness_master:
					p.update(self.fitness_master[str(p['param'])])
				else:
					p.update(self.clac_score(p['param']))
					print('')
				fitness.append(p)
			else:
				fitness.append(p)

		# All Generation fitness
		for fit in fitness:
			param = fit['param']
			self.fitness_master[str(param)] = {k:v for k,v in fit.items() if k!='param'}

		# This generation fitness
		df = pd.DataFrame(fitness)
		df = df.sort(['score0', 'score1'], ascending=[False, True])
		fitness = df.to_dict('records')

		return fitness


	def mutate(self, parent):
		idx = int(math.floor(random.random()*len(parent)))
		child = copy.deepcopy(parent)
		child[idx] = random.choice(self.param_ranges[idx])

		return child


	def crossover(self, parent1, parent2):
		if CROSSOVER_TYPE == 'two-point':
			length = len(parent1)
			r1 = int(math.floor(random.random()*length))
			r2 = r1 + int(math.floor(random.random()*(length-r1)))

			child = copy.deepcopy(parent1)
			child[r1:r2] = parent2[r1:r2]
		
		elif CROSSOVER_TYPE == 'uniform':
			sample = random.sample(range(len(parent1)), int(len(parent1)/2))
			sample.sort()

			child = copy.deepcopy(parent1)
			for s in sample:
				child[s] = parent2[s]

		return child


	def debug(self, gen, fitness):
		print('')
		print('############################## Generation %2s ##############################' % str(gen))
		print('')
		print(pd.DataFrame(fitness)[['score0', 'score1', 'param']].to_string())
		print('')
		print('BEST: Test Accuracy: %s, Time Cost: %s' % (round(fitness[0]['score0'], 6), round(fitness[0]['score1'], 6)))
		params = Param().convert_param(fitness[0]['param'])
		for p in sorted(params):
			print('%-12s:'%p, params[p])
		print('')

		# Log top fitness each generation.
		top_fitness = fitness[0]
		top_fitness.update({'gen': gen})
		self.log_top.append(top_fitness)
		df = pd.DataFrame(self.log_top)
		df.to_csv(LOG_DIR+LOG_FILE_TOP, index=False)


if __name__ == "__main__":
	GA().main()

