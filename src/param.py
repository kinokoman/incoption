# coding: utf-8

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
import random

from data_fizzbuzz import DataFizzBuzz

reload(sys)
sys.setdefaultencoding("utf-8")


class Param:
    def __init__(self):
        pass


    def main(self):
        param = []
        
        param.append(random.randint(0, 3))  # 0: Output Weight
        param.append(random.randint(0, 3))  # 1: Output Standard deviation
        param.append(random.randint(0, 1))  # 2: Output Bias
        param.append(random.randint(0, 0))  # 3: Output Activation Function
        param.append(random.randint(0, 1))  # 4: Train Optimaize
        param.append(random.randint(0, 3))  # 5: Learning Rate
        param.append(random.randint(0, 2))  # 6: Batch Size
        param.append(random.randint(0, 4))  # 7: The Number of Iteration
        param.append(random.randint(0, 3))  # 8: The Number of hidden layer
        
        # 9~: Hidden Layer Design
        idx = 8
        for i in range(param[-1]):
            param.append(random.randint(0, 2))  # The Number of Node
            param.append(random.randint(0, 3))  # Output Weight
            param.append(random.randint(0, 3))  # Output Standard Deviation
            param.append(random.randint(0, 1))  # Output Bias
            param.append(random.randint(0, 3))  # Output Activation Function

        self.convert_param(param)        


    def convert_param(self, param):
        #param = [2, 1, 0, 0, 0, 1, 2, 4, 1, 2, 2, 1, 0, 1]
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
            print '%-12s'%k, dic[k]



if __name__ == "__main__":
    Param().main()


