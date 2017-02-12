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

|Score                                 |FizzBuzz        |MNIST                      |
|:-------------------------------------|---------------:|--------------------------:|
|Test Accuracy                         |100.0 %         |98.2 %                     |
|Time Cost                             |101 minutes     |976 minutes                |
|**Parameter**                         |                |                           |
|The Number of Hidden Layer            |1               |1                          |
|The Number of Batch                   |10              |10                         |
|The Number of Epoch                   |1000            |100                        |
|1st Hidden Layer's Activation Function|tanh            |relu                       |
|1st Hidden Layer's Bias               |zeros           |zeros                      |
|1st Hidden Layer's The Number of Node |100             |100                        |
|1st Hidden Layer's Standard Deviation |0.1             |0.001                      |
|1st Hidden Layer's Weight             |trancated normal|trancated normal           |
|Output Activation Fuction             |-               |-                          |
|Output Bias                           |zeros           |ones                       |
|Output Standard Deviation             |0.001           |0.01                       |
|Output Weight                         |ones            |ones                       |
|Training Optimaization Function       |Adam Optimizer  |Grandient Descent Optimizer|
|Training Rate                         |0.1             |0.1                        |


## Requirement
I have confirmed that it can be used on Windows and Mac.
Linux is not confirmed, but I don't think it will be useless.

### Windows
- NVIDIA GeForce 1080 x1
- Windows 10 Home 64bit
- Python 3.5.2
- [Anaconda 4.2.0 64-bit](https://www.continuum.io/downloads)
- [tensorflow 0.12.0](https://www.tensorflow.org/get_started/os_setup)

The PC with NVIDIA GeForce 1080 which I have is only Windows 10 and TensorFlow on Windows supports only Python 3.

### Mac
- Intel Core i7 2.9GHz
- macOS Sierra
- Python 2.7.10
- [tensorflow 0.8.0](https://www.tensorflow.org/get_started/os_setup)

Incpotion worked on Mac when I fixed the following code in `deeplearnig.py`.

```python
# deeplearnig.py
def train_network(self, data, network, params):
    ...
    sess.run(tf.global_variables_initializer())  # Comment this for Mac
    #tf.initialize_all_variables().run()  # Uncomment this for Mac.
```


## Usage
If you execute it as is, it starts to design the model to solve the MNIST problem, and you leave it.

```
> cd {YOUR-PATH}\incoption\src
> python incoption.py
```

When finished, the best score, Deep Learning parameters and time cost are displayed.

```
BEST: Test Accuracy: 0.982, Time Cost: 9.552549

batch_size  : 10                        # The number of Batch
h1_activ    : relu                      # 1st Hidden Layer's Activation Function
h1_bias     : zeros                     # 1st Hidden Layer's Bias
h1_n_node   : 100                       # 1st Hidden Layer's The Number of Node
h1_stddev   : 0.001                     # 1st Hidden Layer's Standard Deviation
h1_weight   : truncated_normal          # 1st Hidden Layer's Weight
n_h_layer   : 100                       # The Number of Layer
n_iter      : 1                         # The Number of Epoch
o_activ     :                           # Output Activation Fuction
o_bias      : ones                      # Output Bias
o_stddev    : 0.01                      # Output Standard Deviation
o_weight    : ones                      # Output Weight
tr_opt      : GradientDescentOptimizer  # Training Optimaization Function
tr_rate     : 0.1                       # Training Rate

Took 973 minutes.
```

If you want to change the parameters or the learning target, edit the following part of `incoption.py`.
Passing to `self.data` is the order of `[trian_data, train_label, test_data, test_label]`.
See `data_mnist.py` for details.

```python
# incoption.py
N_HIDDEN_LAYER = 1  # The Number of Hidden layer

N_POP = 40          # Population
N_GEN = 25          # The Number of Generation
MUTATE_PROB = 0.5   # Mutation probability
ELITE_RATE = 0.25   # Elite rate

DEBUG = True

class Incoption:
    def __init__(self):
        #self.data = DataFizzBuzz().main()
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
