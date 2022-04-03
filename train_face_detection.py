#!/usr/bin/env python3
# Script to train and test a neural network with TF's Keras API for face detection

import os
import sys
import argparse
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D

# This file contains the training data
def load_data_from_npz_file(file_path):
    """
    Load data from npz file
    :param file_path: path to npz file with training data
    :return: input features and target data as numpy arrays
    """
    data = np.load(file_path)
    return data['input'], data['target']

# Mean and variance centering data
def normalize_data_per_row(data):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data as a numpy array with dimensions NxHxWxC
    :return: normalized data with pixel values in [0,1] (array with same dimensions as input)
    """

    # sanity checks!
    assert len(data.shape) == 4, "Expected the input data to be a 4D matrix"

    return data / 255

# Split into train and test 
def split_data(input, target, train_percent):
    """
    Split the input and target data into two sets
    :param input: inputs [NxM] matrix
    :param target: target [Nx1] matrix
    :param train_percent: percentage of the data that should be assigned to training
    :return: train_input, train_target, test_input, test_target
    """
    assert input.shape[0] == target.shape[0], \
        "Number of inputs and targets do not match ({} vs {})".format(input.shape[0], target.shape[0])

    indices = list(range(input.shape[0]))
    np.random.shuffle(indices)

    num_train = int(input.shape[0]*train_percent)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    return input[train_indices, :], target[train_indices,:], input[val_indices,:], target[val_indices,:]

# Build the model
def build_nonlinear_model():
    """
    Build NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    
    model = Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    model.summary()
    
    return model

def train_model(model, train_input, train_target, val_input, val_target,
                epochs=200, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """

    norm_train_input = n2(train_input)
    norm_val_input = n2(val_input)

    # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 loss='binary_crossentropy',
                 metrics=['binary_accuracy'])
                 
     # tensorboard callback
    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

     # save checkpoint callback
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'best_weights.h5'),
                                                            monitor='binary_accuracy',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            save_freq=1)

    # do training for the specified number of epochs and with the given batch size
    # TODO - Add callbacks to fit funciton
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
          validation_data=(norm_val_input, val_target),
          callbacks=[tbCallBack, checkpointCallBack])



def main(npz_data_file, batch_size, epochs, lr, val, logs_dir, build_fn=build_nonlinear_model):
    """
    Main function that performs training and test on a validation set
    :param npz_data_file: npz input file with training data
    :param batch_size: batch size to use at training time
    :param epochs: number of epochs to train for
    :param lr: learning rate
    :param val: percentage of the training data to use as validation
    :param logs_dir: directory where to save logs and trained parameters/weights
    """

    input, target = load_data_from_npz_file(npz_data_file)
    N = input.shape[0]
    assert N == target.shape[0], \
        "The input and target arrays had different amounts of data ({} vs {})".format(N, target.shape[0]) # sanity check!
    print("Loaded {} training examples.".format(N))

    train_input, train_target, val_input, val_target = split_data(input, target, val)
    model = build_fn()
    train_model(model, train_input, train_target, val_input, val_target, epochs=epochs, learning_rate=lr, batch_size=batch_size)

def n2(data):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data as a numpy array with dimensions NxHxWxC
    :return: normalized data with pixel values in [0,1] (array with same dimensions as input)
    """

    # sanity checks!
    assert len(data.shape) == 4, "Expected the input data to be a 4D matrix"

    if np.max(data) > 255:
        normalized_data = data / 255
    else:
        normalized_data = data
    return normalized_data


if __name__ == "__main__":

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="number of epochs for training",
                        type=int, default=50)
    parser.add_argument("--batch_size", help="batch size used for training",
                        type=int, default=100)
    parser.add_argument("--lr", help="learning rate for training",
                        type=float, default=1e-3)
    parser.add_argument("--val", help="percent of training data to use for validation",
                        type=float, default=0.8)
    parser.add_argument("--input", help="input file (npz format)",
                        type=str, required=True)
    parser.add_argument("--logs_dir", help="logs directory",
                        type=str, default="")
    parser.add_argument("--load_model", help="path to the model",
                    type=str, default="")
    args = parser.parse_args()

    if len(args.logs_dir) == 0: # parameter was not specified
        args.logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))

    if not os.path.isdir(args.logs_dir):
        os.makedirs(args.logs_dir)

    if len(args.load_model) > 0:
        build_fn = lambda : tf.keras.models.load_model(args.load_model, compile=False)
    else:
        build_fn = build_nonlinear_model

    # run the main functio
    main(args.input, args.batch_size, args.epochs, args.lr, args.val, args.logs_dir, build_fn=build_fn)
    sys.exit(0)
