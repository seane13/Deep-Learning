# Deep-Learning
LSTM Stock Predictor

## Instructionns
In this assignment, you will use deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price.


## imports/ installs
import numpy as np
import pandas as pd
import hvplot.pandas
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

## Conclusion

The stock price model was much closer to predicting actual price than the fng model. 