import numpy as np
import matplotlib.pyplot as plt
import dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error


def data_preparation(data, seq_l):
    # exact number of samples and its feature dimension
    data = np.reshape(data, (data.shape[0], 1))
    print "type", type(data)
    print data.shape
    n_samples, n_features = data.shape[0], data.shape[1]
    # create input sequences for lstm w.r.t. the given sequence length
    X = list()
    y = list()
    for i in range(0, n_samples-seq_l, 1):
        sample = data[i:i+seq_l]
        X.append(sample)
        y.append(data[i+seq_l])

    # convert input into a 2D array
    X = np.array(X)  # suitable dimension for the lstm input
    y = np.array(y)

    return X, y


def fit_lstm(X,y, test_X, test_y, batch_size, n_epochs, n_neurons):
    # dim(X) = (n_samples, time steps for a sequence, n_features of a sample)
    # y: predicted value of next time step

    # generate a lstm model
    # keep the state (hidden layer value) inside lstm, which means the states computed for one batch will be used as
    # initial state for next batch
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True, stateful=True))
    model.add(LSTM(units=4))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    tra_loss = []
    val_loss = []
    # training the model for a given number of epochs(iterations on a dataset)
    # at the beginning of each epoch need to reset the state (parameters of hidden layers)
    for i in range(n_epochs):
        history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1,validation_data=(test_X,test_y), shuffle=False)
        tra_loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        model.reset_states()

    return model, tra_loss, val_loss


if __name__ == "__main__":
    total_time_range = 2004 # dataset contains (total_time_range-seq_l) many data samples
    with_noise = False
    std = 3
    period = 400
    mode = 5
    smooth = False
    ratio = 0.05
    n_epochs = 1000
    n_neurons = 9
    batch_size = 100
    test_size = 1000
    seq_length = 4

    x,y = dataset.dataset_with_trend_and_season(total_time_range, with_noise, period)

    # adding anomalies in both datasets separately
    n_a_train = ratio*(total_time_range-seq_length-test_size)
    n_a_test = ratio*test_size
    y_with_a, train_a, test_a, train_time, test_time = dataset.dataset_add_anomaly(x,y,test_size,n_a_train,n_a_test,std)

    # smoothing
    s_y = dataset.holt_winter_smoothing(np.copy(y_with_a), period)
    #s_y = dataset.linear_regression(np.copy(y_with_a), window=400)
    print "size of org data: ", len(y_with_a)
    print "size of smooted data: ", len(s_y)

    plt.title("Holt Winter smoothing")
    plt.plot(y_with_a[-500:], 'b', label='org data')
    plt.plot(s_y[-500:], 'xkcd:orange', label='smoothed data')
    plt.legend()
    plt.show()

    # preparing data sets for lstm
    d_X, d_y = data_preparation(s_y, seq_length)
    print "d_y shape", d_y.shape
    print "d_X shape", d_X.shape
    # splitting data set in train and test
    train_X, train_y = d_X[0:-test_size], d_y[0:-test_size]
    test_X, test_y = d_X[-test_size:], d_y[-test_size:]

    # fit the model with training data
    lstm_model, tra_loss, val_loss = fit_lstm(train_X, train_y, test_X, test_y, batch_size=batch_size,n_epochs=n_epochs, n_neurons=n_neurons)
    y_pred_train = lstm_model.predict(train_X, batch_size=batch_size, verbose=1)
    rmse_train = mean_squared_error(train_y, y_pred_train)
    print ('Train MSE: %.3f' % rmse_train)
    # make batch prediction
    y_pred = lstm_model.predict(test_X, batch_size=batch_size, verbose=1)
    # for i in range(test_y.shape[0]):
        # print ('Excepted=%.3f, Predicted=%.3f' % (test_y[i], y_pred[i]))
    rmse =mean_squared_error(test_y, y_pred)
    print ('Test MSE: %.3f' % rmse)
    """

    # export model
    model_json = lstm_model.to_json()
    with open("model.json","w") as json_file:
        json_file.write(model_json)

    model_yaml = lstm_model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    """

    plt.plot(tra_loss, 'b', label='traing_loss')
    plt.plot(val_loss, 'g', label='validation_loss')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure()
    plt.title('(smoothed) testset results')
    plt.plot(y_with_a[-test_size:], 'k', label='org data')
    plt.plot(test_y, 'b', label='smoothed data')
    plt.plot(y_pred, 'r', label='prediction')
    #test_time = [test_time[i]-(total_time_range-test_size) for i in range(len(test_time))]
    #test_time = [i for i in test_time if i>0]
    #plt.scatter(test_time, test_a)
    plt.legend(loc='lower right')
    plt.show()

    plt.title('(smoothed) training set results')
    plt.plot(y_with_a[0:1000], 'k', label='org data')
    plt.plot(train_y[0:1000], 'b', label='smoothed data')
    plt.plot(y_pred_train[0:1000], 'r', label='prediction')
    plt.legend(loc='lower right')

    plt.show()

    # predict anomaly based on previous dataset
    delta = [abs(y_pred[i]-y_with_a[i]) for i in range(-test_size,0)]
    a_id = [j+(total_time_range-test_size) for (j, i) in enumerate(delta) if i > 2*std]
    print "predicted anomaly index in test set", np.sort(a_id)
    print "truely anomaly inced in test set", np.sort(test_time)
    print "length predicted anomaly: ", len(a_id)
    print "length true anomaly: ", len(test_time)
    precision = float(len([i for i in range(0, len(a_id)) if a_id[i] in test_time]))/float(len(test_time))
    recall = float(len([i for i in range(0, len(a_id)) if not a_id[i] in test_time]))/float(len(test_time))
    print "precision: ", precision
    print "recall: ", recall