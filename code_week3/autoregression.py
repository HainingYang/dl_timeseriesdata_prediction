import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import dataset
from statsmodels.tsa.ar_model import AR
from math import sqrt


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
    mse = sqrt(mean_squared_error(pred_y, test_y))
    #print ('Test MSE: %.3f' % mse)
    return mse


def autoRegression(data, test_size, with_a, std, a_x):
    train, test = data[0:-test_size], data[-test_size:]
    p_mse = persistence_error(data, test_size)

    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    print ('Lag variables: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)

    # make prediction
    preds = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    pred_train = model_fit.predict(start=int(model_fit.k_ar), end=len(train)-1, dynamic=False)

    pred_mse = sqrt(mean_squared_error(preds, test))
    train_mse = sqrt(mean_squared_error(pred_train, train[int(model_fit.k_ar):]))
    print ('Train MSE from Autoregression: %.3f' % train_mse)
    print ('Test MSE from Autoregression: %.3f' % pred_mse)
    print ('Test MSE from Persistence Model: %.3f' % p_mse)

    if with_a:
        delta = [abs(preds[i] - test[i]) for i in range(len(preds))]
        a_id = [j for (j,i) in enumerate(delta) if i > std]

        if len(a_id) == 0:
            print "no detected anomalies!"
        else:
            print "detected index of anomalies: ", a_id
            print "ground truth index of anomalies: ", a_x

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


def linerRegression(X, y, test_size, with_a, std, x_a, period, anomalies, seq_l, y_org):
    regr_model = linear_model.LinearRegression()
    X = np.squeeze(X)
    y = np.squeeze(y)

    print X.shape
    print y.shape

    train_X, train_y = X[0:-test_size], y[0:-test_size]
    test_X, test_y = X[-test_size:], y[-test_size:]

    print train_X.shape

    x_a_gt = np.copy(x_a[(train_X.shape[0]/period+1):])
    print "x_a_g before", x_a_gt

    anomalies_gt = anomalies[(train_X.shape[0]/period+1):]
    x_a_gt = [int(i-train_X.shape[0]-seq_l) for i in x_a_gt] # index start from 0

    print "x_a_g", x_a_gt
    print "anomalies_g", anomalies_gt
    #print "test value at anomalies", test_y[x_a_gt]

    regr_model.fit(train_X, train_y)
    pred_t = regr_model.predict(train_X)
    train_mse = sqrt(mean_squared_error(pred_t, train_y))
    print ('Train MSE from Linearregression: %.3f' % train_mse)
    preds = regr_model.predict(test_X)
    t_mse = sqrt(mean_squared_error(preds, test_y))
    print ('Test MSE from Linearregression: %.3f' % t_mse)

    if with_a:
        delta = [abs(preds[i] - test_y[i]) for i in range(len(preds))]
        a_id_pred = [j for (j, i) in enumerate(delta) if i > std]

        if len(a_id_pred) == 0:
            print "no detected anomalies!"
        else:
            print "detected index of anomalies: ", a_id_pred
            print "predicted value at anomalies: ", preds[a_id_pred]
            print "ground truth test value at anomalies: ", test_y[a_id_pred]

    # plot results
    plt.title('linearregression test set prediction')
    plt.plot(test_y, 'b', label='ground truth')
    #plt.scatter(x_a_gt, anomalies_gt, c='g', label='true anomalies')
    plt.plot(preds, 'r', label='prediction')
    plt.legend(loc='lower right')
    plt.show()

    if with_a:
        plt.plot(test_y, 'g', label='ground truth')
        a = np.array(np.arange(0, len(test_y)))
        y_org = np.squeeze(y_org)
        y_org = y_org[-test_size:]
        plt.fill_between(a, y1=y_org + std, y2=y_org - std, facecolor="green", alpha=0.5)
        plt.scatter(x_a_gt, anomalies_gt, c='g', label='true anomalies')
        plt.scatter(a_id_pred, test_y[a_id_pred], c='r', label='predicted anomalies')
        plt.legend(loc='lower right')
        plt.show()


if __name__ == '__main__':
    total_time_range = 20007
    with_noise = False
    anomaly_interval = 200
    std = 3
    period = 200
    smooth = False
    mode = 6

    test_size = 2000

    d = dataset.dataset_selection(mode, total_time_range, with_noise, period, anomaly_interval, std, smooth)

    X, y = d["x"], d["y"]

    seq_l = 4
    with_a = False
    x_a = []
    anomalies = []
    y_org = []
    if mode % 2 == 0:
        with_a = True
        x_a = d["x_a"]
        anomalies = d["a"]
        y_org = d["y_org"]
        y_org = np.reshape(y_org, (total_time_range, 1))
        _, y_org = data_preparation(y_org, seq_l)

    # run autoregression model
    #autoRegression(y, test_size, with_a, std, x_a)

    # run linear regression model

    data = np.reshape(y, (total_time_range, 1))

    d_X, d_y = data_preparation(data, seq_l)

    linerRegression(d_X, d_y, test_size, with_a, std, x_a, period, anomalies, seq_l,y_org )

