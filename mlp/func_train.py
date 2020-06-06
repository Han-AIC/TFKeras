# File: func_train.py
# Used by: model_1dConv_traffic.py, model_FCNN_Traffic.py
# Function: Provide model compilation and training functions for brevity.

# Import Dependencies

from tensorflow.keras.optimizers import SGD, Adam

# Tensorflow compile function
def compile_model(model,
                learning_rate,
                momentum,
                loss_function,
                metrics):

    model.compile(optimizer=SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True), loss=loss_function, metrics=metrics)
    return model

# Tensorflow fit function
def train_model(model,
              trainX,
              trainY,
              testX,
              testY,
              callbacks,
              epoch_Num,
              batch_size):

      model_History = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epoch_Num, batch_size = batch_size, callbacks = callbacks, verbose=True)
      return model_History
