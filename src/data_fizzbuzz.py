# coding: utf-8

import sys
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf-8')

BORDER = 101
NUM_DIGITS = 10


class DataFizzBuzz:
    def __init__(self):
        pass


    def main(self):
        # Train
        train_data = np.array([self.binary_encode(i, NUM_DIGITS) for i in range(BORDER, 2**NUM_DIGITS)])
        train_label = np.array([self.fizz_buzz_encode(i) for i in range(BORDER, 2**NUM_DIGITS)])
        
        """
        for i in range(14):
            train_data = np.append(train_data, train_data, axis=0)
            train_label = np.append(train_label, train_label, axis=0)
        """
        
        # Test
        test_data = np.array([self.binary_encode(i, NUM_DIGITS) for i in range(0, BORDER)])
        test_label = np.array([self.fizz_buzz_encode(i) for i in range(0, BORDER)])

        # Collect
        data = [train_data, train_label, test_data, test_label]

        return data


    def binary_encode(self, i, num_digit):
        binary = np.array([i >> d & 1 for d in range(NUM_DIGITS)])

        return binary


    def fizz_buzz_encode(self, i):
        if i % 15 == 0:
            result = np.array([0, 0, 0, 1])
        elif i % 5 == 0:
            result = np.array([0, 0, 1, 0])
        elif i % 3 == 0:
            result = np.array([0, 1, 0, 0])
        else:
            result = np.array([1, 0, 0, 0])

        return result


if __name__ == "__main__":
    DataFizzBuzz().main()

