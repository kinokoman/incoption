# coding: utf-8

import random
import math
import copy
import operator
import pandas as pd

N_ITEMS = 20
N_POP = 30
N_GEN = 25
MUTATE_PROB = 0.3
ELITE_RATE = 0.5

class GA:
    def __init__(self):
        self.items = {}
        self.fitness_master = {}


    def main(self): 
        print('Generation  1:'), 
        pop = [{'param': p} for p in self.get_population()]
        fitness = self.evaluate(pop)
        print(fitness[0]['score0'], fitness[0]['score1'], fitness[0]['param'])

        for g in range(N_GEN-1):
            print('Generation %2s:' % str(g+2)),
            
            # Get elites
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
            print(fitness[0]['score0'], fitness[0]['score1'], fitness[0]['param'])

            
    def get_population(self):
        # Make items
        for i in xrange(N_ITEMS):
            self.items[i] = (random.randint(0, 100), random.randint(1, 10))  # value, weight

        # Make population
        pop = []
        for i in range(N_POP):
            ind = [self.items[k] for k in random.sample(range(N_ITEMS), 5)]
            pop.append(ind)

        return pop


    def clac_score(self, indivisual):
        dic = {}
        dic['score0'] = 0  # value
        dic['score1'] = 0  # weight
        for ind in indivisual:
            dic['score0'] += ind[0]
            dic['score1'] += ind[1]
            
        return dic


    def evaluate(self, pop):
        fitness = []
        for p in pop:
            if not p.has_key('score0'):
                # The indivisual made by crossover or mutation existed before
                if self.fitness_master.has_key(str(p['param'])):
                    p.update(self.fitness_master[str(p['param'])])
                # The indivisual is the first
                else:
                    p.update(self.clac_score(p['param']))
                fitness.append(p)
            else:
                fitness.append(p)

        # Save fitness to all genaration dictinary
        for fit in fitness:
            param = fit['param']
            self.fitness_master[str(param)] = {k:v for k,v in fit.items() if k!='param'}

        # This generation fitness
        df = pd.DataFrame(fitness)
        df = df.sort(['score0', 'score1'], ascending=[False, True])
        fitness = df.to_dict('records')
        
        #print(df)
        
        return fitness


    def mutate(self, parent):
        ind_idx = int(math.floor(random.random()*len(parent)))
        item_idx = random.choice(range(N_ITEMS))
        child = copy.deepcopy(parent)
        child[ind_idx] = self.items[item_idx]

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


