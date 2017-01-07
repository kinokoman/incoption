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
        pass


    def generate_random_seq_nums(self):
        seq_nums = []
        
        seq_nums.append(random.randint(0, 3))  # 0: Output Weight
        seq_nums.append(random.randint(0, 3))  # 1: Output Standard deviation
        seq_nums.append(random.randint(0, 1))  # 2: Output Bias
        seq_nums.append(random.randint(0, 0))  # 3: Output Activation Function
        seq_nums.append(random.randint(0, 1))  # 4: Train Optimaize
        seq_nums.append(random.randint(0, 3))  # 5: Learning Rate
        seq_nums.append(random.randint(0, 2))  # 6: Batch Size
        seq_nums.append(random.randint(0, 4))  # 7: The Number of Iteration
        seq_nums.append(random.randint(1, 1))  # 8: The Number of hidden layer
        
        # 9~: Hidden Layer Design
        idx = 8
        for i in range(seq_nums[-1]):
            seq_nums.append(random.randint(0, 2))  # The Number of Node
            seq_nums.append(random.randint(0, 3))  # Output Weight
            seq_nums.append(random.randint(0, 3))  # Output Standard Deviation
            seq_nums.append(random.randint(0, 1))  # Output Bias
            seq_nums.append(random.randint(0, 3))  # Output Activation Function

        return seq_nums


    def convert_param(self, seq_nums):
        params = {}

        # 0: Output Weight
        if   seq_nums[0] == 0: params['o_weight'] = 'zeros'
        elif seq_nums[0] == 1: params['o_weight'] = 'ones'
        elif seq_nums[0] == 2: params['o_weight'] = 'random_normal'
        elif seq_nums[0] == 3: params['o_weight'] = 'truncated_normal'

        # 1: Output Standard deviation
        if   seq_nums[1] == 0: params['o_stddev'] = 0.1
        elif seq_nums[1] == 1: params['o_stddev'] = 0.01
        elif seq_nums[1] == 2: params['o_stddev'] = 0.001
        elif seq_nums[1] == 3: params['o_stddev'] = 0.0001
        
        # 2: Output Bias
        if   seq_nums[2] == 0: params['o_bias'] = 'zeros'
        elif seq_nums[2] == 1: params['o_bias'] = 'ones'
        
        # 3: Output Activation Function
        if   seq_nums[3] == 0: params['o_activ'] = ''

        # 4: Train Optimaize
        if   seq_nums[4] == 0: params['tr_opt'] = 'GradientDescentOptimizer'
        elif seq_nums[4] == 1: params['tr_opt'] = 'AdamOptimizer'
        
        # 5: Learning Rate
        if   seq_nums[5] == 0: params['tr_rate'] = 0.1
        elif seq_nums[5] == 1: params['tr_rate'] = 0.01
        elif seq_nums[5] == 2: params['tr_rate'] = 0.001
        elif seq_nums[5] == 3: params['tr_rate'] = 0.0001
                
        # 6: Batch Size
        if   seq_nums[6] == 0: params['batch_size'] = 10
        elif seq_nums[6] == 1: params['batch_size'] = 50
        elif seq_nums[6] == 2: params['batch_size'] = 100
        
        # 7: The Number of Iteration
        if   seq_nums[7] == 0: params['n_iter'] = 1
        elif seq_nums[7] == 1: params['n_iter'] = 10
        elif seq_nums[7] == 2: params['n_iter'] = 100
        elif seq_nums[7] == 3: params['n_iter'] = 1000
        elif seq_nums[7] == 4: params['n_iter'] = 10000

        # 8: The Number of hidden layer
        if   seq_nums[8] == 0: params['n_h_layer'] = 0
        elif seq_nums[8] == 1: params['n_h_layer'] = 1
        elif seq_nums[8] == 2: params['n_h_layer'] = 2
        elif seq_nums[8] == 3: params['n_h_layer'] = 3
        
        # 9~: Hidden Layer Design
        idx = 8
        for i in range(params['n_h_layer']):
            # The Number of Node
            if   seq_nums[idx+1+5*i] == 0: params['h%s_n_node'%(i+1)] = 10
            elif seq_nums[idx+1+5*i] == 1: params['h%s_n_node'%(i+1)] = 50
            elif seq_nums[idx+1+5*i] == 2: params['h%s_n_node'%(i+1)] = 100

            # Output Weight
            if   seq_nums[idx+2+5*i] == 0: params['h%s_weight'%(i+1)] = 'zeros'
            elif seq_nums[idx+2+5*i] == 1: params['h%s_weight'%(i+1)] = 'ones'
            elif seq_nums[idx+2+5*i] == 2: params['h%s_weight'%(i+1)] = 'random_normal'
            elif seq_nums[idx+2+5*i] == 3: params['h%s_weight'%(i+1)] = 'truncated_normal'

            # Output Standard deviation
            if   seq_nums[idx+3+5*i] == 0: params['h%s_stddev'%(i+1)] = 0.1
            elif seq_nums[idx+3+5*i] == 1: params['h%s_stddev'%(i+1)] = 0.01
            elif seq_nums[idx+3+5*i] == 2: params['h%s_stddev'%(i+1)] = 0.001
            elif seq_nums[idx+3+5*i] == 3: params['h%s_stddev'%(i+1)] = 0.0001
            
            # Output Bias
            if   seq_nums[idx+4+5*i] == 0: params['h%s_bias'%(i+1)] = 'zeros'
            elif seq_nums[idx+4+5*i] == 1: params['h%s_bias'%(i+1)] = 'ones'
            
            # Output Activation Function
            if   seq_nums[idx+5+5*i] == 0: params['h%s_activ'%(i+1)] = ''
            elif seq_nums[idx+5+5*i] == 1: params['h%s_activ'%(i+1)] = 'relu'
            elif seq_nums[idx+5+5*i] == 2: params['h%s_activ'%(i+1)] = 'tanh'
            elif seq_nums[idx+5+5*i] == 3: params['h%s_activ'%(i+1)] = 'softmax'

        # Dudeg
        for k in sorted(params):
            print '%-12s'%k, params[k]
        print

        return params



if __name__ == "__main__":
    Param().main()


