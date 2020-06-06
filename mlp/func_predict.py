# File: func_predict.py
# Used by: model_1dConv_traffic.py, model_FCNN_Traffic.py
# Function: Plot training and prediction metrics, write post-training reports.

# Import Dependencies

import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

# Plot training loss and validation loss together - Spot overfitting if curves begin to diverge.
def plot_training_loss(model_history):
    plt.plot(model_history.history['loss'], label='loss')
    plt.plot(model_history.history['val_loss'], label='val_loss')
    plt.legend()


# Plot test set predictions against targets, to determine goodness of model fit.
def plot_predictions_traffic_volume(model, X_test, y_test):

    test_predictions = model.predict(X_test).flatten()

    # Set fig size limits slightly beyond min/max values in predictions.
    minVal = min([val.min() for val in [test_predictions,y_test]]) * 1.1
    maxVal = max([val.max() for val in [test_predictions,y_test]]) * 1.1

    plt.scatter(y_test, test_predictions)
    plt.xlabel('Target Traffic Volume')
    plt.ylabel('Predicted Traffic Volume ')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(minVal, maxVal)
    plt.ylim(minVal, maxVal)
    _ = plt.plot([-100, 100], [-100, 100])

# Write network params and loss performance to csv file for FCNN model (No conv layers)
def write_report_FCNN(model_report_name,
                    batch_size,
                    epoch_Num,
                    input_shape,
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
                    model_val_mae):

    with open(model_report_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Network: ", model_report_name])
        writer.writerow(["Training Batch Size: ", batch_size])
        writer.writerow(["Number of Epochs: ", epoch_Num])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Losses"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Val Loss: ", model_val_loss])
        writer.writerow(["Val MSE: ", model_val_mse])
        writer.writerow(["Val MAE: ", model_val_mae])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Architecture (Input/Output)"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Input Vector Size: ", input_shape])
        writer.writerow(["Output Vector Size: ", outputSize])
        writer.writerow(["Output Activation: ", output_activation])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Architecture (Dense)"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Input Vector Size: ", input_shape])
        writer.writerow(["Output Vector Size: ", outputSize])
        writer.writerow(["Initial Dense Layer Width: ", denseWidth])
        writer.writerow(["Number of Dense Layers: ", denseLength])
        writer.writerow(["Dense growth/shrinkage factor: ", denseGrowth])
        writer.writerow(["Dense Activation: ", activation_function])
        writer.writerow(["Dropout Value: ", dropout_val])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Learning Hyper Params"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Momentum: ", momentum])
        writer.writerow(["Loss Function: ", loss_function])
        writer.writerow(["Learning Rate: ", learning_rate])
    csvfile.close()

# Write network params and loss performance to csv file for 1dConv model
def write_report_1dConv(model_report_name,
                        batch_size,
                        epoch_Num,
                        input_shape,
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
                        output_activation,
                        momentum,
                        loss_function,
                        learning_rate,
                        model_val_loss,
                        model_val_mse,
                        model_val_mae):

    with open(model_report_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Network: ", model_report_name])
        writer.writerow(["Training Batch Size: ", batch_size])
        writer.writerow(["Number of Epochs: ", epoch_Num])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Losses"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Val Loss: ", model_val_loss])
        writer.writerow(["Val MSE: ", model_val_mse])
        writer.writerow(["Val MAE: ", model_val_mae])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Architecture (Input/Output)"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Input Vector Size: ", input_shape])
        writer.writerow(["Output Vector Size: ", outputSize])
        writer.writerow(["Output Activation: ", output_activation])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Architecture (Dense)"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Initial Dense Layer Width: ", denseWidth])
        writer.writerow(["Number of Dense Layers: ", denseLength])
        writer.writerow(["Dense Growth/Shrinkage factor: ", denseGrowth])
        writer.writerow(["Dense Activation: ", activation_function])
        writer.writerow(["Dropout Value: ", dropout_val])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Architecture (Conv)"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Initial Number of Conv Filters: ", convFilters])
        writer.writerow(["Number of Conv Layers: ", convLength])
        writer.writerow(["Conv Growth/Shrinkage factor: ", convGrowth])
        writer.writerow(["Conv Filter size: ", convFilterSize])
        writer.writerow(["Conv Pooling size: ", poolSize])
        writer.writerow(["Conv Padding size: ", padding])
        writer.writerow(["Conv Activation: ", activation_function])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Network Learning Hyper Params"])
        writer.writerow(["--------------------------------"])
        writer.writerow(["Momentum: ", momentum])
        writer.writerow(["Loss Function: ", loss_function])
        writer.writerow(["Learning Rate: ", learning_rate])
    csvfile.close()
