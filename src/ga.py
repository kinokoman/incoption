# coding: utf-8

import random
import math
import copy
import operator
import pandas as pd

from param import Param
from deeplearning import DeepLearning

N_ITEMS = 20
N_POP = 30 #20
N_GEN = 25 #25
MUTATE_PROB = 0.1
ELITE_RATE = 0.5

N_HIDDEN_LAYER = 0


class GA:
    def __init__(self):
        self.param_ranges = Param().get_param_ranges(N_HIDDEN_LAYER)
        self.fitness_master = {}


    def main(self): 
        pop = [{'param': p} for p in self.get_population()]
        print('population')
        for p in pop:
            print(p)
        print
        
        for g in range(N_GEN):
            print('Generation%3s:' % str(g))#, 

            # Get elites
            fitness = self.evaluate(pop)
            elites = fitness[:int(len(pop)*ELITE_RATE)]

            # Cross and mutate
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
            
            # Evaluate indivisual
            fitness = self.evaluate(pop)
            pop = fitness[:]

            """
            print
            for fit in fitness:
                print(fit)
            """
            print(pop[0]['score0'], pop[0]['score1'], pop[0]['param'])
            print

            
    def get_population(self):
        # Make population
        pop = []
        for i in range(N_POP):
            ind = Param().make_param(N_HIDDEN_LAYER)
            pop.append(ind)

        return pop


    def clac_score(self, indivisual):
        test_accuracy, time_cost = DeepLearning().main2(indivisual)
            
        dic = {}
        dic['score0'] = test_accuracy
        dic['score1'] = time_cost

        print(dic)
        
        return dic


    def evaluate(self, pop):
        fitness = []
        for p in pop:
            if not 'score0' in p:
                if str(p['param']) in self.fitness_master:
                    print('Yes, No')
                    p.update(self.fitness_master[str(p['param'])])
                else:
                    #print('clac_score!')
                    print('No, No')
                    p.update(self.clac_score(p['param']))
                fitness.append(p)
            else:
                print('Yes, Yes')
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
        length = len(parent1)
        r1 = int(math.floor(random.random()*length))
        r2 = r1 + int(math.floor(random.random()*(length-r1)))
        
        child = copy.deepcopy(parent1)
        child[r1:r2] = parent2[r1:r2]

        return child


if __name__ == "__main__":
    GA().main()


