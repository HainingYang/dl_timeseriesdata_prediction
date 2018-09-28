import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from mpmath import sqrt
from statsmodels.tsa.ar_model import AR


def data_generator(start, time_steps):
    x = np.array(range(start, time_steps))
    noise = np.random.uniform(-0.2, 0.2, time_steps)
    y = np.sin(np.pi * x / 50) + np.cos(np.pi * x / 25)+noise
    #plt.show()
    return y


# generate a graph of Xt and Xt-1 to see the correlation
def check_corelation(data):
    s = pd.Series(data)
    lag_plot(s)
    plt.show()


# compute mse of test set using a persistence model as the worst performance for a model
def persistence_error(data, test_size):
    test_X = data[-test_size: -1]  # data at time t-1
    test_y = data[-test_size+1:]  # data at time t
    print len(test_y)
    print len(test_X)
    assert len(test_X) == len(test_y)

    # use persistence model: f(x)=x
    pred_y = test_X
    # compute mse of the persistence model
    mse = mean_squared_error(pred_y, test_y)
    #print ('Test MSE: %.3f' % mse)
    return mse


def autoRegression(data, test_size):
    train, test = data[0:-test_size], data[-test_size:]
    p_mse = persistence_error(data, test_size)

    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    print ('Lag variables: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    #make prediction
    preds = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    pred_train = model_fit.predict(start=int(model_fit.k_ar), end=len(train)-1, dynamic=False)

    pred_mse = mean_squared_error(preds, test)
    train_mse = mean_squared_error(pred_train, train[int(model_fit.k_ar):])
    print ('Train MSE from Autoregression: %.3f' % train_mse)
    print ('Test MSE from Autoregression: %.3f' % pred_mse)
    print ('Test MSE from Persistence Model: %.3f' % p_mse)
    # plot results
    plt.title('autoregression test set prediction')
    plt.plot(test, 'b', label='ground truth')
    plt.plot(preds, 'r', label='prediction')
    plt.legend(loc='lower right')
    plt.show()


def data_preparation(data, seq_l):
    # exact number of samples and its feature dimension
    n_samples, n_features = data.shape[0], data.shape[1]
    # create input sequences for lstm w.r.t. the given sequence length
    X = list()
    y = list()
    for i in range(0, n_samples-seq_l, 1):
        sample = data[i:i+seq_l]
        X.append(sample)
        y.append(data[i+seq_l])

    # convert input into a 2D array
    X = np.array(X)
    y = np.array(y)

    return X, y


def linerRegression(X, y, test_size):
    regr_model = linear_model.LinearRegression()
    X = np.squeeze(X)
    y = np.squeeze(y)
    train_X, train_y = X[0:-test_size], y[0:-test_size]
    test_X, test_y = X[-test_size:], y[-test_size:]
    regr_model.fit(train_X, train_y)
    pred_t = regr_model.predict(train_X)
    train_mse = mean_squared_error(pred_t, train_y)
    print ('Train MSE from Linearregression: %.3f' % train_mse)
    preds = regr_model.predict(test_X)
    t_mse = mean_squared_error(preds, test_y)
    print ('Test MSE from Linearregression: %.3f' % t_mse)
    plt.title('linearregression test set prediction')
    plt.plot(test_y, 'b', label='ground truth')
    plt.plot(preds, 'r', label='prediction')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    total_time_range = 100000
    test_size = 1000

    data = data_generator(0, total_time_range)
    autoRegression(data, test_size)

    seq_l = 7
    data = np.reshape(data, (total_time_range, 1))
    d_X, d_y = data_preparation(data, seq_l)
    linerRegression(d_X, d_y, test_size)