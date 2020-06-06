# File: func_1dConv.py
# Used by: model_1dConv_traffic.py
# Function: Initialize tf.keras.models.Model according to hyperparameter specification in
#           model_1dConv_traffic.py

# Import Dependencies

import tensorflow as tf
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

#############################################################
#### Function for constructing one dimensional CNN model ####

# Layer declaration is in tensorflow functional form.

def build_model_CNN(input_shape,
                    outputSize,
                    denseWidth,
                    denseLength,
                    denseGrowth,
                    convFilters,
                    convLength,
                    convGrowth,
                    convFilterSize,
                    poolSize,
                    padding,
                    dropout_val,
                    activation_function,
                    output_activation):

    # Specify input layer dimensions
    i = Input(shape=(1, input_shape[2],))

    # Build convolutional layers
    for j in range(convLength):

      # First conv block takes input layer i
      if j == 0:
        x = Conv1D(convFilters, (convFilterSize), activation=activation_function, padding=padding)(i)
        x = BatchNormalization()(x)
        x = Conv1D(convFilters, (convFilterSize), activation=activation_function, padding=padding)(x)
        x = AveragePooling1D((poolSize), padding=padding,strides=1)(x)

      # Subsequent conv block takes x
      else:
        x = Conv1D(convFilters, (convFilterSize), activation=activation_function, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Conv1D(convFilters, (convFilterSize), activation=activation_function, padding=padding)(x)
        x = AveragePooling1D((poolSize), padding=padding,strides=1)(x)

      # Modify number of conv filters in next layer according to conv growth factor (convGrowth)
      convFilters = convFilters * convGrowth

    # Global Average Pooling rather than max pooling *may* give better results on non-image data.
    x = GlobalAveragePooling1D()(x)

    # Build dense layers
    for k in range(denseLength):

      x = Dense(denseWidth, activation=activation_function, kernel_regularizer=l2(0.001))(x)
      x = Dropout(dropout_val)(x)
      denseWidth = denseWidth * denseGrowth

    # Specify output layer with linear activation for regression task.
    x = Dense(outputSize, activation=output_activation)(x)

    # Declare model
    model = Model(i, x)

    return model
