# coding: utf-8

######################
#       Common       #
######################
DATA = 'mnist'                       # Select from 'fizzbuzz', 'mnist'
LOG_DIR = '../log/'                  # Path to logging


######################
#  Genetic Algrithm  #
######################
N_HIDDEN_LAYER = 2                   # The Number of Hidden layer, make this a value range not a value!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

N_POP = 40                           # Population
N_GEN = 25                           # The Number of Generation
MUTATE_PROB = 0.5                    # Mutation probability
ELITE_PROB = 0.25                    # Elite probability

DEBUG_GA = True                      # Debug for Genetic Algorithm
LOG_FILE_TOP = 'log_top.csv'         # Log file for top of each generation
LOG_FILE_DETAIL = 'log_detail.csv'   # Log file for detail logging


######################
#   Deep Learning    #
######################
DEBUG_DL = True                      # Debug for Deep Learning
LOG_TRAIN = False                    # Log for each training, make this one of method parameter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
LOG_FILE_TRAIN = 'log_train.csv'     # Log file for each training

