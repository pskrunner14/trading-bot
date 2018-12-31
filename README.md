# Overview

Stock Trading Bot using Deep Reinforcement Learning (Deep Q-learning), Keras and TensorFlow. Reinforcement Learning is a type of machine learning that allows us to create intelligent agents that learn from the environment by interacting with it, it learns by trial and error. After each action, the agent receives the feedback in the form of reward or penalty. The feedback consists of the reward and next state of the environment. The reward is usually defined by a human. We can define reward as the profit from selling the stock bought at the original starting point.

![Google Stock Trading bot](./extra/visualization.png)

Trading `GOOGL`, 2018 with a profit of $517.44

## Dataset

You can either use the Historical Financial data already present under `data/` or download your own from [Yahoo! Finance](https://ca.finance.yahoo.com/) for training.

## Getting Started

In order to train the model, you will need to install the required python packages:

```bash
pip install -r requirements.txt
```

Now you can open up a terminal and start training the agent:

```bash
python train.py --train-stock data/GOOGL.csv --val-stock data/GOOGL_2018.csv
```

Once you're done training, run the evaluation script and let the agent make stock decisions:

```bash
python evaluate.py --eval-data data/AAPL_2018.csv --window-size 10 --model-name model_GOOGL_10 --debug
```

Now you are all set up!

## References

* [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)

## Built With

* Python
* TensorFlow
* NumPy

## Credits

The project is a modification/impovement over [edwardhdlu/q-trader](https://github.com/edwardhdlu/q-trader).
