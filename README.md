# Lagrangian Neural Networks


## Getting Started

First ensure that you have [python3 with pip (a package manager for python) installed](https://www.python.org/downloads/).

```
pip install argparse numpy pygame tqdm torch
```

Then you can run:

```
python simulator.py
```


## Training

You can train a Neural Network (classical or Lagrangian) on a variety of systems. To get started training a Lagrangian NN, run:

```
python train.py --experiment double_pendulum
```

To view more training options:
```
python train.py --help
```


## Simulator

You can view the results of the model you trained (or simply just the analytical solution for the behavior of a system). To run the simulator, you can run:

```
python simulator.py --experiment double_pendulum
```

Note that you do not need to have trained a model to get this to work.

To view more training options:

```
python simulator.py --help
```
