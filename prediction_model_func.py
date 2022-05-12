import numpy as np
import pandas as pd
import pickle
import time
import random
from sklearn.metrics import confusion_matrix
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import pickle
import os
import pdb
from PIL import Image

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
AUTOTUNE = tf.data.AUTOTUNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Dropout 
from tensorflow.keras.layers import Flatten, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

AUTOTUNE = tf.data.AUTOTUNE
tf.get_logger().setLevel('INFO')
tf.random.set_seed(2266)


def gen_train_test_data(df, ticker, date_split, ls_x, num_pca, response='response_win', add_pca=False):
    mask = df['ticker'] == ticker
    df_sample =  df[mask].copy().reset_index(drop=True)
    
    # split between train and test
    df_train = df_sample.loc[df_sample['date'] <= date_split, 
                                 ls_x + ['response', 'response_win', 'date', 'ticker']].reset_index(drop=True)
    df_test = df_sample.loc[df_sample['date'] > date_split, 
                                ls_x + ['response', 'response_win', 'date', 'ticker']].reset_index(drop=True)
    
    if add_pca:
        df_train, df_test = add_pca_component_top(df_train, df_test, ls_x, num_pca)
        ls_x = ['pca_%d' % (x) for x in range(num_pca)]
        
    X_train = df_train[ls_x]
    X_test = df_test[ls_x]
    y_train = df_train[response]
    y_test = df_test[response]
    return df_train, df_test, X_train, X_test, y_train, y_test


def add_pca_component_top(df_train, df_test, ls_x, num_pca):
    pca = PCA()
    ls_pca = ['pca_%d' % (x) for x in range(num_pca)]
    
    df_pca = pca.fit_transform(df_train[ls_x])
    df_train[ls_pca] = df_pca[:,:num_pca]
    
    df_pca = pca.transform(df_test[ls_x])
    df_test[ls_pca] = df_pca[:,:num_pca]
    return df_train, df_test

def linear_regression_model(X_train, y_train):
    # Initialize a Linear Regression model
    model = LinearRegression()

    # Fit the linear model on the train data
    model.fit(X_train, y_train)
    return model

def summarize_model_output(model, X_train, X_test, y_train, y_test, verbose=True):
    # Predict the response variable for the test set using the trained model
    y_pred_train = model.predict(X_train).reshape(-1,)
    y_pred_test = model.predict(X_test).reshape(-1,)

    # Compute the MSE for the test data
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    if verbose:
        #scores
        print(f"   ~~~Train/Test MSE: {mse_train:.4f}/{mse_test:.4f}")
        print(f"   ~~~Train/Test R2: {r2_train:.4f}/{r2_test:.4f}")
        
    dc_sum = {'mse_train':  mse_train, 
              'mse_test':   mse_test, 
              'r2_train':   r2_train, 
              'r2_test':    r2_test}
    
    return dc_sum, y_pred_test

def residual_coefficient_plot(model, y_pred, y_test, X_test, ticker, method, lookback):
    residuals = y_pred - y_test

    #Bar plot of coefficient values 
    coefs = pd.concat([pd.DataFrame(X_test.columns),pd.DataFrame(np.transpose(model.coef_))], axis = 1)
    coefs.columns = ['Coefficients', 'Values']

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('%s - %s (Lookback %d)' % (method, ticker, lookback), fontsize=20)
    
    ax = axs[0]
    ax.plot(y_pred, residuals,'.',color='#faa0a6', markersize=6, label="Residuals")
    ax.hlines(0, -1, 1)
    ax.set_title('Residual Plot', fontsize=19)
    ax.set_xlabel("Forward 1 Week Return", fontsize=16)
    ax.set_ylabel("Residuals", fontsize=16)

    ax = axs[1]
    ax.bar(coefs['Coefficients'], coefs['Values'], label="Residuals")
    ax.set_title('Coefficients of the Multilinear Model', fontsize=19)
    ax.set_xlabel("Principal Components", fontsize=16)
    ax.set_ylabel("Coefficient values", fontsize=16)
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    
def plot_loss(model_history, ticker, method, lookback, start=0, out_file = None):
    """
    This helper function plots the NN model accuracy and loss.
    Arguments:
        model_history: the model history return from fit()
        out_file: the (optional) path to save the image file to.
    """
    history = model_history
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 4))
    fig.suptitle('%s - %s (Lookback %d)' % (method, ticker, lookback), fontsize=20)
        
    ax[0].plot(history.history['mse'][start:])
    ax[0].plot(history.history['val_mse'][start:])
    ax[0].set_title('Model MSE')
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'validation'], loc='upper left')
    
    # summarize history for loss
    ax[1].plot(history.history['loss'][start:])
    ax[1].plot(history.history['val_loss'][start:])
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'validation'], loc='upper left')
    fig.tight_layout()
    
    if out_file:
        fig.savefig(out_file)
        
def simple_ffnn_model(input_size, num_layer, kernel_size, activation, is_batchnorm, is_dropout, 
                      task_type, dropout, is_summary=False):
    # input layer
    inputs = Input(shape=(input_size,))
    
    # first hidden layer
    x = Dense(kernel_size[0], activation=activation)(inputs)
    if is_batchnorm:
        x = BatchNormalization()(x)
    if is_dropout:
        x = Dropout(dropout)(x)
    
    # additional hidden layer
    if len(kernel_size) > 1:
        for size in kernel_size[1:]:
            x = Dense(kernel_size[0], activation=activation)(inputs)
            if is_batchnorm:
                x = BatchNormalization()(x)
            if is_dropout:
                x = Dropout(dropout)(x)        
    
    # output layer
    outputs = Dense(1, activation=task_type)(x)
    
    # model
    model = Model(inputs=inputs, outputs=outputs, name='simple_ffnn')
    if is_summary:
        FFNN.summary()
    return model


def fit_nn_model(model, X_train, y_train, optimizer, loss, metrics, validation_split, batch_size, epochs, plot_history=True):
    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # fit model
    history = model.fit(X_train, y_train, 
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0)

    return model, history


def fc_layer(x, n_nodes, dropout_rate=None, activation='relu'):
    x = Dense(n_nodes, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)    
    return x

def conv_block(x, filters, kernel_size, stride_size, pool_size, padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides = stride_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x = MaxPooling2D(pool_size=pool_size, padding=padding)(x)
    return x


def conv_architect(inputs, num_cblock, filters_size, kernel_size, pool_size, stride_size, fc_node=100, fc_drop = 0.5, padding='same'):
    
    if (num_cblock != len(filters_size)) | (num_cblock < 1):
        print("Incorrect filter setting.")
        return -1
    
    # convolution layer
    for i in range(num_cblock):
        if i == 0 :
            x = conv_block(inputs, filters=filters_size[i], kernel_size=kernel_size, stride_size=stride_size, pool_size = pool_size)
        else:
            x = conv_block(x, filters=filters_size[i], kernel_size=kernel_size, stride_size =(1,1),  pool_size = pool_size)       
    
    # flatten layer
    x = Flatten()(x)
    
    x = fc_layer(x, fc_node, fc_drop)

    # output layer
    output = Dense(1,activation='linear')(x)
    
    return output

def compile_model_result(dc_model, is_plot=True):
    # put all model results together
    df_model = pd.DataFrame()
    df_forecast = pd.DataFrame()
    for model, result in dc_model.items():
        df_temp = dc_model[model]['sum'].reset_index().rename(columns={'index': 'item'})
        df_temp['model'] = model
        df_model = df_model.append(df_temp, ignore_index=True)

        df_temp = dc_model[model]['pred']
        df_temp['model'] = model
        df_forecast = df_forecast.append(df_temp, ignore_index=True)
    return df_model, df_forecast