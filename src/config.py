# coding: utf-8

######################
#       Common       #
######################
DATA = 'fizzbuzz'                    # Select from 'fizzbuzz', 'mnist'
LOG_DIR = '../log/'                  # Log directory path


######################
#  Genetic Algrithm  #
######################
N_HIDDEN_LAYER = 1                   # The Number of Hidden layer, make this a value range not a value!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

N_POP = 4 #40                           # Population
N_GEN = 2 #25                           # The Number of Generation
MUTATE_PROB = 0.5                    # Mutation probability
ELITE_PROB = 0.25                    # Elite probability

DEBUG_GA = True                      # Debug for Genetic Algorithm
LOG_FILE_TOP = 'log_top.csv'         # Log file for top of each generation
LOG_FILE_DETAIL = 'log_detail.csv'   # Log file for detail logging


######################
#   Deep Learning    #
######################
DEBUG_DL = True                      # Debug for Deep Learning
MODEL_DIR = '../model/'              # Model directory path
MODEL_NAME = 'model.ckpt'            # Model name
LOG_FILE_TRAIN = 'log_train.csv'     # Log file for each training

