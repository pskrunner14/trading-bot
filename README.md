# Overview

This project implements a Stock Trading Bot, trained using Deep Reinforcement Learning, specifically Deep Q-learning. Implementation is kept simple and as close as possible to the algorithm discussed in the paper, for learning purposes.

## Introduction

Generally, Reinforcement Learning is a family of machine learning techniques that allow us to create intelligent agents that learn from the environment by interacting with it, as they learn an optimal policy by trial and error. This is especially useful in many real world tasks where supervised learning might not be the best approach due to various reasons like nature of task itself, lack of appropriate labelled data, etc.

## Approach

This work uses a Model-free Reinforcement Learning technique called Deep Q-Learning (neural variant of Q-Learning).
At any given time (episode), an agent abserves it's current state (n-day window stock price representation), selects and performs an action (buy/sell/hold), observes a subsequent state, receives some reward signal (difference in portfolio position) and lastly adjusts it's parameters based on the gradient of the loss computed.
The important idea here is that this technique can be applied to any real world task that can be described loosely as a Markovian process.

## Results

Trained on `GOOG` 2010-17 stock data, tested on 2019 with a profit of $1141.45 (validated on 2018 with profit of $863.41):

![Google Stock Trading episode](./extra/visualization.png)

You can obtain similar visualizations of your model evaluations using the [notebook](./visualize.ipynb) provided.

## Some Caveats

- At any given state, the agent can only decide to buy/sell one stock at a time. This is done to keep things as simple as possible as the problem of deciding how much stock to buy/sell is one of portfolio redistribution.
- The n-day window feature representation is a vector of subsequent differences in Adjusted Closing price of the stock we're trading followed by a sigmoid operation, done in order to normalize the values to the range [0, 1].
- Training is prefferably done on CPU due to it's sequential manner, after each episode of trading we replay the experience and update model (Q-function) parameters.

## Data

You can download Historical Financial data from [Yahoo! Finance](https://ca.finance.yahoo.com/) for training, or even use some sample datasets already present under `data/`.

## Getting Started

In order to use this project, you'll need to install the required python packages:

```bash
pip3 install -r requirements.txt
```

Now you can open up a terminal and start training the agent:

```bash
python3 train.py data/GOOG.csv data/GOOG_2018.csv
```

Once you're done training, run the evaluation script and let the agent make trading decisions:

```bash
python3 eval.py data/GOOG_2019.csv --model-name model_GOOG_50 --debug
```

Now you are all set up!

## References

- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/)

## Credits

I'd also like to point out that this project is a modification/impovement over [edwardhdlu/q-trader](https://github.com/edwardhdlu/q-trader). Key differences include model architecture, loss function, optimizer and a faster training procedure as well some minor improvements in operational sementics.
