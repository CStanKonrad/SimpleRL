# SimpleRL

## Repo content
* very simple examples of Q-learning
* gym problems have been changed to boost learning speed

## Environment
* tensorflow `2.0.0-rc0`
* numpy `1.17.2`

## Usage
* `python3 cartpole.py`
* if you want to train model you should change `cartpole(True)` to `cartpole(False)` in `cartpole.py`

## Issues
* usage of `model.predict(dataset)` seems to cause memory leak 
https://github.com/keras-team/keras/issues/13118 https://github.com/tensorflow/tensorflow/issues/30324
