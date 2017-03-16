# coding: utf-8

import sys
from collections import Counter
import random


class Param:
    def __init__(self):
        pass


    def main(self):
        pass


    def make_param(self, n_hidden_layer):
        ranges = self.get_param_ranges(n_hidden_layer)
        param = [random.choice(r) for r in ranges]

        return param


    def get_param_ranges(self, L):
        ranges = []
        
        ranges.append(range(0, 3+1))  # 0: Output Weight
        ranges.append(range(0, 3+1))  # 1: Output Standard deviation
        ranges.append(range(0, 1+1))  # 2: Output Bias
        ranges.append(range(0, 0+1))  # 3: Output Activation Function
        ranges.append(range(0, 1+1))  # 4: Train Optimaize
        ranges.append(range(0, 3+1))  # 5: Learning Rate
        ranges.append(range(0, 2+1))  # 6: Batch Size
        ranges.append(range(0, 2+1))  # 7: The Number of Epoch, exists till 3!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ranges.append(range(L, L+1))  # 8: The Number of hidden layer
        
        # 9~: Hidden Layer Design
        for i in range(L):
            ranges.append(range(0, 2+1))  # The Number of Node
            ranges.append(range(0, 3+1))  # Output Weight
            ranges.append(range(0, 3+1))  # Output Standard Deviation
            ranges.append(range(0, 1+1))  # Output Bias
            ranges.append(range(0, 3+1))  # Output Activation Function

        return ranges


    def convert_param(self, param):
        params = {}

        # 0: Output Weight
        if   param[0] == 0: params['o_weight'] = 'zeros'
        elif param[0] == 1: params['o_weight'] = 'ones'
        elif param[0] == 2: params['o_weight'] = 'random_normal'
        elif param[0] == 3: params['o_weight'] = 'truncated_normal'

        # 1: Output Weight Standard deviation
        if   param[1] == 0: params['o_stddev'] = 0.1
        elif param[1] == 1: params['o_stddev'] = 0.01
        elif param[1] == 2: params['o_stddev'] = 0.001
        elif param[1] == 3: params['o_stddev'] = 0.0001
        
        # 2: Output Bias
        if   param[2] == 0: params['o_bias'] = 'zeros'
        elif param[2] == 1: params['o_bias'] = 'ones'
        
        # 3: Output Activation Function
        if   param[3] == 0: params['o_activ'] = ''

        # 4: Train Optimaize
        if   param[4] == 0: params['tr_opt'] = 'GradientDescentOptimizer'
        elif param[4] == 1: params['tr_opt'] = 'AdamOptimizer'
        
        # 5: Learning Rate
        if   param[5] == 0: params['tr_rate'] = 0.1
        elif param[5] == 1: params['tr_rate'] = 0.01
        elif param[5] == 2: params['tr_rate'] = 0.001
        elif param[5] == 3: params['tr_rate'] = 0.0001
                
        # 6: The number of Batch
        """
        if   param[6] == 0: params['n_batch'] = 10
        elif param[6] == 1: params['n_batch'] = 50
        elif param[6] == 2: params['n_batch'] = 100
        """
        if   param[6] == 0: params['n_batch'] = 0.01
        elif param[6] == 1: params['n_batch'] = 0.05
        elif param[6] == 2: params['n_batch'] = 0.10
        
        # 7: The Number of Epoch
        if   param[7] == 0: params['n_epoch'] = 1
        elif param[7] == 1: params['n_epoch'] = 10
        elif param[7] == 2: params['n_epoch'] = 100
        elif param[7] == 3: params['n_epoch'] = 1000   # Not use now

        # 8: The Number of Hidden Layer
        if   param[8] == 0: params['n_h_layer'] = 0
        elif param[8] == 1: params['n_h_layer'] = 1
        elif param[8] == 2: params['n_h_layer'] = 2
        elif param[8] == 3: params['n_h_layer'] = 3
        
        # 9~: Hidden Layer Design
        idx = 8
        for i in range(params['n_h_layer']):
            # The Number of Node
            if   param[idx+1+5*i] == 0: params['h%s_n_node'%(i+1)] = 10
            elif param[idx+1+5*i] == 1: params['h%s_n_node'%(i+1)] = 50
            elif param[idx+1+5*i] == 2: params['h%s_n_node'%(i+1)] = 100

            # Hidden Layer Weight
            if   param[idx+2+5*i] == 0: params['h%s_weight'%(i+1)] = 'zeros'
            elif param[idx+2+5*i] == 1: params['h%s_weight'%(i+1)] = 'ones'
            elif param[idx+2+5*i] == 2: params['h%s_weight'%(i+1)] = 'random_normal'
            elif param[idx+2+5*i] == 3: params['h%s_weight'%(i+1)] = 'truncated_normal'

            # Hidden Layer Weight Standard deviation
            if   param[idx+3+5*i] == 0: params['h%s_stddev'%(i+1)] = 0.1
            elif param[idx+3+5*i] == 1: params['h%s_stddev'%(i+1)] = 0.01
            elif param[idx+3+5*i] == 2: params['h%s_stddev'%(i+1)] = 0.001
            elif param[idx+3+5*i] == 3: params['h%s_stddev'%(i+1)] = 0.0001
            
            # Hidden Layer Bias
            if   param[idx+4+5*i] == 0: params['h%s_bias'%(i+1)] = 'zeros'
            elif param[idx+4+5*i] == 1: params['h%s_bias'%(i+1)] = 'ones'
            
            # Hidden Layer Activation Function
            if   param[idx+5+5*i] == 0: params['h%s_activ'%(i+1)] = ''
            elif param[idx+5+5*i] == 1: params['h%s_activ'%(i+1)] = 'relu'
            elif param[idx+5+5*i] == 2: params['h%s_activ'%(i+1)] = 'tanh'
            elif param[idx+5+5*i] == 3: params['h%s_activ'%(i+1)] = 'softmax'

        return params



if __name__ == "__main__":
    Param().main()


