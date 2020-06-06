# File: func_FCNN.py
# Used by: model_FCNN_traffic.py
# Function: Initialize tf.keras.models.Model according to hyperparameter specification in
#           model_FCNN_traffic.py

# Import Dependencies

import tensorflow as tf
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model


#############################################################
#### Function for constructing basic FCNN model ####

# Layer declaration is in tensorflow functional form.

def build_model_FCNN(input_shape,
                    outputSize,
                    denseWidth,
                    denseLength,
                    denseGrowth,
                    dropout_val,
                    activation_function,
                    output_activation):
                    
    # Specify input layer dimensions
    i = Input((input_shape[1], ))

    # Initialize dense layers
    for k in range(denseLength):

      # First dense block takes input layer i
      if k == 0:
        x = Dense(denseWidth, activation=activation_function, kernel_regularizer=l2(0.001))(i)
        x = Dropout(dropout_val)(x)

      # Subsequent dense blocks takes input layer x
      else:
        x = Dense(denseWidth, activation=activation_function, kernel_regularizer=l2(0.001))(x)
        x = Dropout(dropout_val)(x)

      # Modify dense layer width according to growth/shrinkage factor (denseGrowth)
      denseWidth = denseWidth * denseGrowth

    # Initialize output layer
    x = Dense(outputSize, activation=output_activation)(x)

    # Declare model
    model = Model(i, x)

    return model
