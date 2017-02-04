Incoption
====

`Incoptin` designs (Deep) Neural Network model automatically and efficiently.

![pic](inception.jpg)

[https://en.wikipedia.org/wiki/Inception](https://en.wikipedia.org/wiki/Inception)

`Incoption` is inspired by the movie "Inception".
Inception means Deep Neural Network and `Incoption` includes the string "opti" of optimization.


## Description
`Incoption` optimizes Deep Learning parameters with Genetic Algorithm.
In order to achieve high accuracy, it designs the Deep Neural Network model automatically and efficiently.
But now it's **only FizzBuzz problem** 100%.


## Requirement
I have confirmed that it can be used on Windows and Mac.
Linux is not confirmed, but I don't think it will be useless.

### Windows
- NVIDIA GeForce 1080 x1
- Windows 10 Home 64bit
- Python 3.5.2
- [Anaconda 4.2.0 64-bit](https://www.continuum.io/downloads)
- [tensorflow 0.12.0](https://www.tensorflow.org/get_started/os_setup)

The PC with NVIDIA GeForce 1080 which I have is only Windows 10, so I use Python 3.
Although it was a CPU, it worked on Mac when I fixed the following code in `deeplearnig.py`.

### Mac
- Intel Core i7 2.9GHz
- macOS Sierra
- Python 2.7.10
- [tensorflow 0.8.0](https://www.tensorflow.org/get_started/os_setup)

```python
# deeplearnig.py
def train_network(self, data, network, params):
    ...
    sess.run(tf.global_variables_initializer())  # Comment this for Mac
    #tf.initialize_all_variables().run()  # Uncommment this for Mac.
```


## Usage
If you execute it as is, it starts to design the model to solve the FizzBuzz problem, so you leave it.

```sh
> cd {YOUR-PATH}\incoption\src
> python incoption.py
```

When finished, the best score, its Deep Learning parameters and the time taken are displayed.

```
BEST: Test Accuracy: 1.0, Time Cost: 1.567682
batch_size  : 10                  # Batch Size
h1_activ    : tanh                # 1st Hidden Layer's Activation Function
h1_bias     : zeros               # 1st Hidden Layer's Bias
h1_n_node   : 100                 # 1st Hidden Layer's The Number of Node
h1_stddev   : 0.1                 # 1st Hidden Layer's Standard Deviation
h1_weight   : truncated_normal    # 1st Hidden Layer's Weight
n_h_layer   : 1                   # The Number of Layer
n_iter      : 1000                # The Number of Epoch
o_activ     :                     # Output Activation Fuction
o_bias      : zeros               # Output Bias
o_stddev    : 0.001               # Output Standard Deviation
o_weight    : ones                # Output Weight
tr_opt      : AdamOptimizer       # Training Optimaization Function
tr_rate     : 0.1                 # Learning Rate

Took 101 minutes.
```

If you want to change the parameters or the learning target, edit the following part of `incoption.py`.
These parameter settings will be automated in the future.
Passing to `self.data` is the order of `[trian_data, train_label, test_data, test_label]`.
See `data_fizzbuzz.py` for details.

```python
N_HIDDEN_LAYER = 1  # The Number of Hidden layer

N_POP = 40          # Population
N_GEN = 25          # The Number of Generation
MUTATE_PROB = 0.5   # Mutation probability
ELITE_RATE = 0.25   # Elite rate

DEBUG = True

class Incoption:
    def __init__(self):
        self.data = DataFizzBuzz().main()
        ...
```


## Install
```sh
> git https://github.com/iShoto/incoption.git
```


## Contribution
1. Fork it
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create new Pull Request


## Licence
[MIT](https://github.com/iShoto/incoption/blob/master/LICENSE)


## Author
[iShoto](https://github.com/iShoto)
