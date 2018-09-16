import numpy as np
import math
import matplotlib.pyplot as plt
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from math import sqrt


# generate sinus and cosine combined function within a time interval, adding noises with choice
def normal_dataset_generator(time_range, with_noise):
    x = np.array(np.arange(0, time_range))
    noise = np.random.uniform(-0.2, 0.2, time_range)
    y = 10*(np.sin(np.pi * (x-25) / 50) + np.cos(np.pi * (x-25) / 25))+20
    if with_noise:
        y = y + noise
    return x,y


def dataset_with_trend(time_range, with_noise):
    x = np.array(np.arange(0, time_range))
    y = 10*(np.sin(np.pi * (x-25) / 50) + np.cos(np.pi * (x-25) / 25))+20
    noise = np.random.uniform(-0.2, 0.2, time_range)
    if with_noise:
        y = y + noise

    for i in range(0, time_range):
        y[i] += 0.1*i
    return x, y


def dataset_with_trend_and_season(time_range, with_noise, period):
    x = np.array(np.arange(0, time_range))
    y = 10*(np.sin(np.pi * (x-25) / 50) + np.cos(np.pi * (x-25) / 25))+20
    noise = np.random.uniform(-0.2, 0.2, time_range)
    if with_noise:
        y = y + noise

    for i in range(0, time_range):
        y[i] += 0.3 * (i-math.floor(i/period)*period)

    return x, y


def dataset_add_anomaly(x, y, test_size, n_a_train, n_a_test, std):

    # generate random time points for the occurance of anomalies
    dataset_l = len(x)
    print dataset_l-test_size
    print n_a_train
    train_time = random.sample(range(0,dataset_l-test_size),int(n_a_train))
    print "train_time", train_time
    test_time = random.sample(range(dataset_l-test_size, len(x)), int(n_a_test))
    print "test_time", test_time

    train_a = []
    test_a = []

    y_with_a = np.copy(y)

    #add anomalies for training set
    for i in range(len(train_time)):
        p = np.random.random()
        if p > 0.5:
            anomaly = y[int(train_time[i])] + (2.3 + p) * std
        else:
            anomaly = y[int(train_time[i])] - (2.3 + 0.5 + p) * std
        train_a.append(anomaly)

        y_with_a[int(train_time[i])] = anomaly

    #add anomalies for test set
    for i in range(len(test_time)):
        p = np.random.random()
        if p > 0.5:
            anomaly = y[int(test_time[i])] + (2.3 + p) * std
        else:
            anomaly = y[int(test_time[i])] - (2.3 + 0.5 + p) * std
        test_a.append(anomaly)

        y_with_a[int(test_time[i])] = anomaly

    return y_with_a, train_a, test_a, train_time, test_time


def holt_winter_smoothing(y, period):
    model = ExponentialSmoothing(y, trend ="add", damped=False, seasonal="add", seasonal_periods=period)
    model_fit = model.fit()
    y = model_fit.predict(0, len(y)-1)

    return y


def moving_average(y, window):
    y_s = np.copy(y)

    for i in range(window, len(y)):
        y_s[i] = sum(y[i-window:i-1])/window
    assert len(y)==len(y_s)
    return y_s


def linear_regression(y, window):

    # data preparation
    X = list()
    Y = list()
    for i in range(0, len(y)-window, 1):
        sample = y[i:i+window]
        X.append(sample)
        Y.append(y[i+window])

    regr_model = linear_model.LinearRegression()
    regr_model.fit(X,Y)
    pred_t = regr_model.predict(X)
    train_mse = sqrt(mean_squared_error(pred_t, Y))
    print ('Train MSE from Linearregression: %.3f' % train_mse)

    s_y = np.concatenate((y[0:window],pred_t), axis=0)
    return s_y


# visualizing dataset with a certain predefining confidence interval
def visualize_ci(x, y, std):
    plt.fill_between(x, y1=y + std, y2=y-std, facecolor="green", alpha=0.5)
    plt.plot(x, y, color="b")
    plt.show()


def visualize_ci_anomaly(x, y, std, x_a, anomalies):
    plt.fill_between(x, y1=y + std, y2=y - std, facecolor="green", alpha=0.5)
    plt.plot(y, color="b")
    plt.scatter(x_a, anomalies, c='r')
    plt.show()


# mode 1: normal dataset; mode 2: normal dataset + anomalies (smooth/not)
# mode 3: dataset with trend; mode 4: dataset + trend + anomalies (smooth/not)
# mode 5: dataset + trend + season; mode 6: dataset + trend + season + anomalies (smooth/not)
def dataset_selection(mode, time_range, with_noise, period, anomaly_interval, std, smooth):
    if mode == 1:
        x, y = normal_dataset_generator(time_range, with_noise)
        return {"x": x, "y": y}

    if mode == 2:
        x, y = normal_dataset_generator(time_range, with_noise)
        x_a, anomalies, y_with_a = dataset_add_anomaly(x, y, anomaly_interval, std)
        if not smooth:
            return {"x": x, "y": y_with_a, "x_a": x_a, "y_org": y, "a": anomalies}
        else:
            return

    if mode == 3:
        x, y = dataset_with_trend(time_range, with_noise)
        return {"x": x, "y": y}

    if mode == 4:
        x, y = dataset_with_trend(time_range, with_noise)
        x_a, anomalies, y_with_a = dataset_add_anomaly(x, y, anomaly_interval, std)
        if not smooth:
            return {"x": x, "y": y_with_a, "x_a": x_a, "y_org": y, "a": anomalies}
        else:
            return

    if mode == 5:
        x, y = dataset_with_trend_and_season(time_range, with_noise, period)
        return {"x": x, "y": y}

    if mode == 6:
        x, y = dataset_with_trend_and_season(time_range,with_noise, period)
        x_a, anomalies, y_with_a = dataset_add_anomaly(x, y, anomaly_interval, std)
        if not smooth:
            return {"x": x, "y": y_with_a, "x_a": x_a, "y_org": y, "a": anomalies}
        else:
            return


if __name__ == "__main__":
    time_range = 5000
    std = 3
    with_noise = False
    anomaly_interval = 40
    period = 100

    x, y = dataset_with_trend_and_season(time_range, with_noise, period)
    test_size = 200
    n_a_train = (time_range-test_size)*0.05
    n_a_test = test_size*0.05
    y_with_a, train_a, test_a, train_time, test_time = dataset_add_anomaly(x,y,test_size,n_a_train,n_a_test,std)

    s3_y = linear_regression(y_with_a, window=50)
    plt.title("Weighted Moving average smoothing")
    plt.plot(y_with_a[1500:2000], 'b-', label='org data')
    plt.plot(s3_y[1500:2000], 'xkcd:orange', label='smoothed data')
    plt.legend(loc='lower right')
    plt.show()


    s1_y = holt_winter_smoothing(np.copy(y_with_a),period)
    s2_y = moving_average(np.copy(y_with_a), window=4)
    s3_y_a = linear_regression(y_with_a, window=50)
    s3_y_b = linear_regression(y_with_a, window=100)
    s3_y_c = linear_regression(y_with_a, window=150)


    rmse1 = sqrt(mean_squared_error(y_with_a, s1_y))
    rmse2 = sqrt(mean_squared_error(y_with_a, s2_y))

    print "RMSE of Holt-winter smoothing is: ", rmse1
    print "RMSE of moving average smoothing is: ", rmse2
    #visualize_ci_anomaly(x,y_with_a,std,train_time+test_time,train_a+test_a)

    plt.title("Holt-Winter smoothing")
    plt.plot(y_with_a[1500:2000], 'b-', label='org data')
    plt.plot(s1_y[1500:2000], 'xkcd:orange', label='smoothed data')
    plt.legend(loc='lower right')
    plt.show()

    plt.title("Moving average smoothing")
    plt.plot(y_with_a[1500:2000], 'b-', label='org data')
    plt.plot(s2_y[1500:2000], 'xkcd:orange',label='smoothed data')
    plt.legend(loc='lower right')
    plt.show()

    plt.title("Weighted Moving average smoothing with window 50")
    plt.plot(y_with_a[1500:2000], 'b-', label='org data')
    plt.plot(s3_y_a[1500:2000], 'xkcd:orange', label='smoothed data')
    plt.legend(loc='lower right')
    plt.show()

    plt.title("Weighted Moving average smoothing with window 100")
    plt.plot(y_with_a[1500:2000], 'b-', label='org data')
    plt.plot(s3_y_b[1500:2000], 'xkcd:orange', label='smoothed data')
    plt.legend(loc='lower right')
    plt.show()

    plt.title("Weighted Moving average smoothing with window 150")
    plt.plot(y_with_a[1500:2000], 'b-', label='org data')
    plt.plot(s3_y_c[1500:2000], 'xkcd:orange', label='smoothed data')
    plt.legend(loc='lower right')
    plt.show()

    plt.title("Comparison of three smoothing methods")
    plt.plot(y_with_a[1500:2000], 'b-')
    plt.plot(s1_y[1500:2000], 'xkcd:coral', label='Holt-Winter')
    plt.plot(s2_y[1500:2000], 'xkcd:orange', label='Moving average')
    plt.plot(s3_y_b[1500:2000], 'xkcd:lime', label='Weighted Moving average with window 100')
    plt.legend(loc='lower middle')
    plt.show()


