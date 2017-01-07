# coding: utf-8

import sys
import importlib
import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Visual:
    def __init__(self):
        pass


    def main(self):
        # Read file
        df = pd.read_csv('./log/acculoss_dropout_1.0_train_0.05_batch_100_iter_10000.csv')
        
        # Set values
        x = df['epoch'].values
        y0 = df['train_loss'].values
        y1 = df['train_accuracy'].values
        y2 = df['test_accuracy'].values

        # Set background color to white
        fig = plt.figure()
        fig.patch.set_facecolor('white')

        # Plot lines
        plt.xlabel('epoch')
        plt.plot(x, y0, label='train_loss')
        plt.plot(x, y1, label='train_accuracy')
        plt.plot(x, y2, label='test_accuracy')
        plt.legend()

        # Visualize
        plt.show()


if __name__ == "__main__":
    Visual().main()

