# coding: utf-8

######################
#       Common       #
######################
DATA = 'mnist'       # Select from 'fizzbuzz', 'mnist'
LOG_PATH = '../log/' # Path to logging


######################
#  Genetic Algrithm  #
######################
N_POP = 40           # Population
N_GEN = 25           # The Number of Generation
MUTATE_PROB = 0.5    # Mutation probability
ELITE_RATE = 0.25    # Elite rate

DEBUG_GA = True      # Debug for Genetic Algorithm


######################
#   Deep Learning    #
######################
N_HIDDEN_LAYER = 2   # The Number of Hidden layer

DEBUG_DL = True      # Debug for Deep Learning
TRAIN_LOG = False    # Log for each training
