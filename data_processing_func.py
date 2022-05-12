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

tf.random.set_seed(2266)

px = 1 / plt.rcParams['figure.dpi']

class TS_Image:
    
    def __init__(self, data, ticker, dir_path, methods=['GADF'], lookback=63, freq=1):
        
        self.lookback = lookback
        self.freq = freq
        self.Sample = {}
        self.methods = methods
        self.ticker = ticker
        self.data = data
        self.dir = os.path.join(dir_path, ticker)
    
    def make_dir(self):
        # make directory
        # ticker directory
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        
        # method directory
        ls_mkdir = [os.path.join(self.dir, method) for method in self.methods]
        for directory in ls_mkdir:
            if not os.path.exists(directory):
                os.mkdir(directory)
                
        # raw image for OHLC
        if 'OHLC' in self.methods:
            OHLC_dir = os.path.join(self.dir, 'OHLC', 'raw')
            if not os.path.exists(OHLC_dir):
                os.mkdir(OHLC_dir)
            
    
    def drawImage(self, df_ts, path):
        
        plt.style.use('dark_background')
        plt.axis('off')
#         sns.set_style("dark")
        
        width = 4 * self.lookback
        
        if self.lookback <= 5:
            height = 36
        elif self.lookback <= 21:
            height = 72
        elif self.lookback <= 63:
            height = 108
        else:
            height = 36*(self.lookback//21)
        #height = width
        
        x = np.arange(0, self.lookback)
        
        
        fig, ax = plt.subplots(1, figsize=(width * px, height * px))
        ax.axis("off")
        for idx, val in df_ts.iterrows():
            # high/low lines
            ax.plot([x[idx], x[idx]], 
                     [val['low'], val['high']], 
                     color='white')
            # open marker
            ax.plot([x[idx], x[idx] - 0.1], 
                     [val['open'], val['open']], 
                     color='white')
            # close marker
            ax.plot([x[idx], x[idx] + 0.1], 
                     [val['close'], val['close']], 
                     color='white')

        ax.plot(df_ts['ma_23d'], color='white');
        ax.plot(df_ts['ma_5d'], color='white');
        ax.set_ylim(self.ax_range);
        ax.set_xticks([]);
        ax.set_yticks([]);
        fig.tight_layout();
        fig.savefig(path,bbox_inches='tight',pad_inches=0);
        fig.clf();
        
        # return numpy array
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        return numpydata
    
    def getImage(self):
        
        self.make_dir()
        self.ls_date = list(self.data['date'].drop_duplicates())[::self.freq]
        
        self.ls_col = [str(x) for x in range(self.lookback - 1, -1, -1)]
        
        ax_min = min(self.data[self.ls_col].max().max(), 1.5)
        ax_max = max(self.data[self.ls_col].min().min(), 0.5)
        
        self.ax_range = (ax_min, ax_max)
        
        for method in self.methods:
            if method == 'GADF':
                self.Sample[method] = self.getGADF()
            if method == 'GASF':
                self.Sample[method] = self.getGASF()
            if method == 'OHLC':
                self.Sample[method] = self.getOHLC()
            if method == 'MTF':
                self.Sample[method] = self.getMTF()

            with open(os.path.join(self.dir, method, 'cnnsample %d.pkl' % (self.lookback)), 'wb') as p:
                pickle.dump(self.Sample[method], p)
                p.close()

        return self.Sample
            
            
    def getOHLC(self):
        self.OHLCSample = []
        
        for dt in self.ls_date:
            # grab time series data 
            df_ts = self.data.loc[self.data.date == dt,self.ls_col+['metric']]
            df_ts = df_ts.T
            df_ts.columns=df_ts.loc['metric']
            df_ts = df_ts[:-1].reset_index()
            df_ts['index']= df_ts['index'].astype('int')
            df_ts = df_ts.sort_values(['index'], ascending = False)
            df_ts['ma_23d'] = df_ts['close'].rolling(window = 23).mean()
            df_ts['ma_5d'] = df_ts['close'].rolling(window = 5).mean()
            
            path = os.path.join(self.dir, 'OHLC', 'raw', dt.strftime("%Y%m%d") + '.png')
            self.OHLCSample.append(self.drawImage(df_ts, path))
            
        return self.OHLCSample
    
    def getGASF(self):
        self.GASFSample = []
        for dt in self.ls_date:
            df_ts = self.data.loc[self.data.date == dt, self.ls_col]
            gasf = GramianAngularField(16, method='summation')
            self.GASFSample.append(gasf.fit_transform(df_ts).T)
            
        return self.GASFSample
    
    def getGADF(self):
        self.GADFSample = []
        for dt in self.ls_date:
            df_ts = self.data.loc[self.data.date == dt, self.ls_col]
            gadf = GramianAngularField(16, method='difference')
            self.GADFSample.append(gadf.fit_transform(df_ts).T)
            
        return self.GADFSample
    
    def getMTF(self):
        self.MTFSample = []
        for dt in self.ls_date:
            df_ts = self.data.loc[self.data.date == dt, self.ls_col]
            mtf = MarkovTransitionField(16)
            self.MTFSample.append(mtf.fit_transform(df_ts).T)
            
        return self.MTFSample
    


def normalize_img(img, label):
    return tf.cast(img, tf.float32)/255.0, label

def gen_tf_dataset(x_set, y_set,batch_size=32):

    ds = tf.data.Dataset.from_tensor_slices((x_set,y_set))
    ds = ds.map(normalize_img, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=ds.cardinality(), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    
    return ds

def get_ytrain(ticker, df_target, date_split, response):
    return df_target.loc[(df_target['ticker'] == ticker) & (df_target['date'] <= date_split), response].values

def get_xtrain(ls_ticker, ticker_index, y_train, dc_image, method):
    X = dc_image[ls_ticker[ticker_index]][method]
    return np.array(X[:y_train[ticker_index].shape[0]])

def get_ytest(ticker, df_target, date_split, response):
    return df_target.loc[(df_target['ticker'] == ticker) & (df_target['date'] > date_split), response].values

def get_xtest(ls_ticker, ticker_index, y_train, dc_image, method):
    X = dc_image[ls_ticker[ticker_index]][method]
    return np.array(X[y_train[ticker_index].shape[0]:])

def gen_train_val_test(dc_image, df_target, date_split, ls_ticker, method, response, val_size=0.33, batch_size = 32):
    
    y_train = [get_ytrain(ticker, df_target, date_split, response) for ticker in ls_ticker]
    x_train = np.concatenate([get_xtrain(ls_ticker, i, y_train, dc_image, method) for i in range(len(ls_ticker))])

    y_test = np.concatenate([get_ytest(ticker, df_target, date_split, response) for ticker in ls_ticker])
    x_test = np.concatenate([get_xtest(ls_ticker, i, y_train, dc_image, method) for i in range(len(ls_ticker))])

    y_train = np.concatenate(y_train)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size)
    
    # generate tf dataset
    ds_train = gen_tf_dataset(x_train, y_train, batch_size=batch_size)
    ds_val = gen_tf_dataset(x_val, y_val, batch_size=batch_size)
    ds_test = gen_tf_dataset(x_test, y_test, batch_size=batch_size)
    
    return ds_train, ds_val, ds_test, (y_train, y_test)

def check_field_coverage(df, ls_ticker, dc_label, date_start, date_end, field):
    df_cov = df.groupby('ticker')[field].agg(['count', 'size']).reset_index()
    df_cov['cov_year'] = df_cov['count'] / 252
    df_cov['cov_pct'] = df_cov['count'] / df_cov['size'] * 100
    df_cov['order'] = df_cov['ticker'].map(dict(zip(ls_ticker, range(len(ls_ticker)))))
    df_cov['label'] = df_cov['ticker'].map(dc_label)
    df_cov = df_cov.sort_values('order').reset_index(drop=True)

    sns.set_style('white')
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('%s Data Coverage from %d to %d' % (field.capitalize(), date_start.year, date_end.year), fontsize=20)

    ax = axs[0]
    df_cov.set_index('ticker')['count'].plot.bar(ax=ax)
    ax.set_title('# of Days', fontsize=15)
    ax.set_ylabel('# of Days', fontsize=13)
    ax.set_xlabel('Asset Class', fontsize=13)
    ax.tick_params(axis='x', rotation=0)

    ax = axs[1]
    df_cov.set_index('ticker')['cov_year'].plot.bar(ax=ax)
    ax.set_title('# of Years', fontsize=15)
    ax.set_ylabel('# of Year', fontsize=13)
    ax.set_xlabel('Asset Class', fontsize=13)
    ax.tick_params(axis='x', rotation=0)

    ax = axs[2]
    df_cov.set_index('ticker')['cov_pct'].plot.bar(ax=ax)
    ax.set_title('Coverage (%)', fontsize=15)
    ax.set_ylabel('Coverage (%)', fontsize=13)
    ax.set_xlabel('Asset Class', fontsize=13)
    ax.tick_params(axis='x', rotation=0)

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    return df_cov
