Incoption
====

`Incoption` designs (Deep) Neural Network model automatically and efficiently.

![pic](inception.jpg)

[https://en.wikipedia.org/wiki/Inception](https://en.wikipedia.org/wiki/Inception)

`Incoption` is inspired by the movie "Inception".
Inception means Deep Neural Network and `Incoption` includes the string "opti" of optimization.


## Description
`Incoption` optimizes Deep Learning parameters with Genetic Algorithm.
In order to achieve high accuracy, it designs the Deep Neural Network model automatically and efficiently.
But it was verified with only FizzBuzz and MNIST.

|Score                               |FizzBuzz        |MNIST                      |
|:-----------------------------------|---------------:|--------------------------:|
|Test Accuracy                       |100.0 %         |98.2 %                     |
|Time Cost                           |101 minutes     |976 minutes                |
|**Parameter**                       |                |                           |
|The Number of Hidden Layer          |1               |1                          |
|The Number of Batch                 |10              |10                         |
|The Number of Epoch                 |1000            |100                        |
|1st Hidden Layer Activation Function|tanh            |relu                       |
|1st Hidden Layer Bias               |zeros           |zeros                      |
|1st Hidden Layer The Number of Node |100             |100                        |
|1st Hidden Layer Standard Deviation |0.1             |0.001                      |
|1st Hidden Layer Weight             |trancated normal|trancated normal           |
|Output Activation Fuction           |-               |-                          |
|Output Bias                         |zeros           |ones                       |
|Output Standard Deviation           |0.001           |0.01                       |
|Output Weight                       |ones            |ones                       |
|Training Optimaization Function     |Adam Optimizer  |Grandient Descent Optimizer|
|Training Rate                       |0.1             |0.1                        |


## Requirement
Incoption works on Windows and Mac.
I don't validate it on Linux but I think that it will works on Linux too.

### Windows
- NVIDIA GeForce 1080 x1
- Windows 10 Home 64bit
- Python 3.5.2
- [Anaconda 4.2.0 64-bit](https://www.continuum.io/downloads)
- [tensorflow 0.12.0](https://www.tensorflow.org/get_started/os_setup)

### Mac
- Intel Core i7 2.9GHz
- macOS Sierra
- Python 2.7.10
- [tensorflow 0.12.1](https://www.tensorflow.org/get_started/os_setup)


## Usage
If you execute it as is, it starts to design the model to solve the MNIST problem, and you leave it.

```
> cd {YOUR-PATH}\incoption\src
> python incoption.py
```

When finished, the best score, Deep Learning parameters and time cost are displayed.

```
BEST: Test Accuracy: 0.982, Time Cost: 9.552549

h1_activ    : relu                      # 1st Hidden Layer Activation Function
h1_bias     : zeros                     # 1st Hidden Layer Bias
h1_n_node   : 100                       # 1st Hidden Layer The Number of Node
h1_stddev   : 0.001                     # 1st Hidden Layer Standard Deviation
h1_weight   : truncated_normal          # 1st Hidden Layer Weight
n_batch     : 10                        # The Number of Batch
n_epoch     : 1                         # The Number of Epoch
n_h_layer   : 100                       # The Number of Layer
o_activ     :                           # Output Activation Fuction
o_bias      : ones                      # Output Bias
o_stddev    : 0.01                      # Output Standard Deviation
o_weight    : ones                      # Output Weight
tr_opt      : GradientDescentOptimizer  # Training Optimaization Function
tr_rate     : 0.1                       # Training Rate

Took 973 minutes.
```

If you want to change the learning target or the parameters, edit `config.py`.

```python
# config.py
######################
#       Common       #
######################
DATA = 'mnist'                       # Select from 'fizzbuzz', 'mnist'
LOG_DIR = '../log/'                  # Log directory path


######################
#  Genetic Algrithm  #
######################
N_HIDDEN_LAYER = 2                   # The Number of Hidden layer

N_POP = 40                           # Population
N_GEN = 25                           # The Number of Generation
MUTATE_PROB = 0.5                    # Mutation probability
ELITE_PROB = 0.25                    # Elite probability
...
```

Give `self.data` training and testing data in order of `[trian_data, train_label, test_data, test_label]`.
See `data_mnist.py` for details.

```python
class Incoption:
	def __init__(self):
		# Data
		if DATA == 'fizzbuzz':
			self.data = DataFizzBuzz().main()
		elif DATA == 'mnist':
			self.data = DataMnist().main()
        ...
```


## Install
```
> cd {YOUR-PATH}
> git clone https://github.com/iShoto/incoption.git
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
[Shoto I.](https://github.com/iShoto)

