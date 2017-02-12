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

N_HIDDEN_LAYER = 1  # The Number of Hidden layer

N_POP = 40          # Population
N_GEN = 25          # The Number of Generation
MUTATE_PROB = 0.5   # Mutation probability
ELITE_RATE = 0.25   # Elite rate

LOG_PATH = '../log/'
DEBUG = True

class Incoption:
    def __init__(self):
        #self.data = DataFizzBuzz().main()
        self.data = DataMnist().main()
        self.param_ranges = Param().get_param_ranges(N_HIDDEN_LAYER)
        self.fitness_master = {}
        self.ga_log = []
        self.calc_log = []  # Temp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
            elites = fitness[:int(N_POP*ELITE_RATE)]

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
        self.calc_log.append({'param': indivisual})
        df = pd.DataFrame(self.calc_log)
        df.to_csv(LOG_PATH+'calc_log.csv', index=False)

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
                fitness.append(p)
            else:
                fitness.append(p)

        # All Generation fitness
        for fit in fitness:
            param = fit['param']
            self.fitness_master[str(param)] = {k:v for k,v in fit.items() if k!='param'}

        # This generation fitness
        df = pd.DataFrame(fitness)
        df = df.sort_values(['score0', 'score1'], ascending=[False, True])
        fitness = df.to_dict('records')

        return fitness


    def mutate(self, parent):
        idx = int(math.floor(random.random()*len(parent)))
        child = copy.deepcopy(parent)
        child[idx] = random.choice(self.param_ranges[idx])

        return child


    def crossover(self, parent1, parent2):
        length = len(parent1)
        r1 = int(math.floor(random.random()*length))
        r2 = r1 + int(math.floor(random.random()*(length-r1)))

        child = copy.deepcopy(parent1)
        child[r1:r2] = parent2[r1:r2]

        return child


    def debug(self, gen, fitness):
        print()
        print('############################## Generation %2s ##############################' % str(gen))
        print()
        print(pd.DataFrame(fitness)[['score0', 'score1', 'param']])
        print()
        print('BEST: Test Accuracy: %s, Time Cost: %s' % (round(fitness[0]['score0'], 6), round(fitness[0]['score1'], 6)))
        params = Param().convert_param(fitness[0]['param'])
        for p in sorted(params):
            print('%-12s:'%p, params[p])
        print()

        # Log top fitness each generation.
        top_fitness = fitness[0]
        top_fitness.update({'gen': gen})
        self.ga_log.append(top_fitness)
        df = pd.DataFrame(self.ga_log)
        df.to_csv(LOG_PATH+'ga_log.csv', index=False)


if __name__ == "__main__":
    Incoption().main()
