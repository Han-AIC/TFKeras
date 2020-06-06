# File: model_FCNN_traffic.py
# Used by: None (Highest level)
# Function: Use all func files to prepare a dataset, build an FCNN model, train it, save it, report it.

# Import Dependencies


from func_data_prep import *
from func_FCNN import *
from func_train import *
from func_predict import *

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping

print("Downloading Data")
data_url = 'https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv'
print("Data Download Complete.")

print("Splitting Data into Train and Test Sets.")
X_train, X_test, y_train, y_test = prep_and_split_traffic(data_url)

print("Complete. Building model to specs listed in model_FCNN_Traffic.py")

# Halting training if val_loss does not improve during any stretch of iterations.
earlyStoppingCallback = EarlyStopping(monitor='val_loss', patience=20)

#############################
### Vital Hyperparameters ###
#############################

# Self-explantory, edit these to alter the model.
outputSize = 1
batch_size = 128
learning_rate = 0.005
momentum = 0.995
loss_function = 'MSE'
metrics = ['mse', 'mae']
epoch_Num = 25
callbacks = []
# early stopping callback is unnecessary for only 25 training epochs.
# callbacks = [earlyStoppingCallback]

# Define dense Layers
# Layers will grow and shrink based on growth factor.
# For example: denseGrowth = 0.5, densewidth = 512
#              First layer is 512, next layer is 256, etc...

# Initial layer width
denseWidth = 1024
# Number of layers
denseLength = 8
# Layer growth factor
denseGrowth = 0.85

# Dropout for additional regularization.
dropout_val = 0.10

# Activation Functions
# Non output activation
activation_function = 'relu'
# Output activation.
output_activation = 'linear'

# Learning rate decay to prevent wild oscillations when closer to minima.
learningRateScheduler = ExponentialDecay(learning_rate,
                                        decay_steps=300,
                                        decay_rate=0.9,
                                        staircase=True)

###### !!!!!!!!!!!!!!! DO NOT EDIT UNLESS YOU KNOW WHAT TO DO !!!!!!!!!!!!!!! ######

model = build_model_FCNN(X_train.shape, outputSize, denseWidth, denseLength, denseGrowth, dropout_val, activation_function, output_activation)
model = compile_model(model, learningRateScheduler, momentum, loss_function, metrics)
model.summary()

print("Complete. Training Commencing.")

model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_Num, batch_size = batch_size, callbacks = callbacks, verbose=True)

###### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ######

# Plot training and validation losses together.
plot_training_loss(model_history)
plt.title('Model Loss Over Epochs', loc = 'center')
plt.show()

# Acquire average of last three losses from model history.
model_val_loss = np.format_float_positional(np.average([model_history.history['val_loss'][-1], model_history.history['val_loss'][-2], model_history.history['val_loss'][-3]]), precision=4, unique=False, fractional=False, trim='k')
model_val_mse = np.format_float_positional(np.average([model_history.history['val_mse'][-1], model_history.history['val_mse'][-2], model_history.history['val_mse'][-3]]), precision=4, unique=False, fractional=False, trim='k')
model_val_mae = np.format_float_positional(np.average([model_history.history['val_mae'][-1], model_history.history['val_mae'][-2], model_history.history['val_mae'][-3]]), precision=4, unique=False, fractional=False, trim='k')

# Trim to four sig figs for brevity.
np.format_float_positional(np.average([model_history.history['val_loss'][-1], model_history.history['val_loss'][-2], model_history.history['val_loss'][-3]]), precision=4, unique=False, fractional=False, trim='k')

print("Average Last 3 val_loss: ", model_val_loss)
print("Average Last 3 val_mse: ", model_val_mse)
print("Average Last 3 val_mae: ", model_val_mae)

# Plot predictions against target labels.
plot_predictions_traffic_volume(model, X_test, y_test)
plt.title('Model Predictions', loc = 'center')
plt.show()

print("Training Complete.")

# Write reports and save the model to a .h5
model_report_name = "./mlp/REPORT_FCNN_traffic_" + str(model_val_loss) + "_" + str(denseWidth) + "_" + str(denseLength) + "_" + str(denseGrowth) + "_" + str(epoch_Num) + ".csv"
model_save_name = "./mlp/TRAINEDMODEL_FCNN_traffic_" + str(model_val_loss) + "_" + str(denseWidth) + "_" + str(denseLength) + "_" + str(denseGrowth) + "_" + str(epoch_Num) + ".h5"
model.save(model_save_name)

write_report_FCNN(model_report_name,
                batch_size,
                epoch_Num,
                X_train.shape,
                outputSize,
                denseWidth,
                denseLength,
                denseGrowth,
                dropout_val,
                activation_function,
                output_activation,
                momentum,
                loss_function,
                learning_rate,
                model_val_loss,
                model_val_mse,
                model_val_mae)

print("See report " + model_report_name + " for model details.")
print("See saved model " + model_save_name + " to load and reuse model.")
