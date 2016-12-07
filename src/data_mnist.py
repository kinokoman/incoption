# coding: utf-8

import sys
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

reload(sys)
sys.setdefaultencoding('utf-8')


class DataMnist:
    def __init__(self):
        pass


    def main(self):
        # Download
        mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

        # Train
        train_data = mnist.train.images
        train_label = mnist.train.labels
 
        # Test
        test_data = mnist.test.images
        test_label = mnist.test.labels

        # Collect
        data = [train_data, train_label, test_data, test_label]

        return data


if __name__ == "__main__":
    DataMnist().main()

