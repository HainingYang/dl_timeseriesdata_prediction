import numpy as np
import matplotlib.pyplot as plt
import dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error


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


def fit_lstm(X,y, test_X, test_y, batch_size, n_epochs, n_neurons):
    # dim(X) = (n_samples, time steps for a sequence, n_features of a sample)
    # y: predicted value of next time step

    # generate a lstm model
    # keep the state (hidden layer value) inside lstm, which means the states computed for one batch will be used as
    # initial state for next batch
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
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
    total_time_range = 10004
    with_noise = False
    anomaly_interval = 200
    batch_size = 100
    test_size = 1000
    seq_length = 4
    std = 3
    period = 400
    mode = 5
    smooth = False

    n_epochs = 700
    n_neurons = 7

    d = dataset.dataset_selection(mode, total_time_range, with_noise, period, anomaly_interval, std, smooth)
    x,y = d["x"], d["y"]

    #x_a, anomalies, y_with_a = dataset.dataset_add_anomaly(x,y,anomaly_interval,std)
    #dataset.visualize_ci_anomaly(x[:500],y[:500],std, x_a[:3], anomalies[:3])
    #plt.plot(y_with_a[:1000])
    #plt.show()
    y= np.reshape(y, (total_time_range, 1))

    d_X, d_y = data_preparation(y, seq_length) # 2000 groups data in total
    train_X, train_y = d_X[0:-test_size], d_y[0:-test_size] # 1900 groups for training
    test_X, test_y = d_X[-test_size:], d_y[-test_size:] # 100 groups for testing

    # fit the model with training data
    lstm_model,tra_loss, val_loss = fit_lstm(train_X, train_y, test_X, test_y, batch_size=batch_size,n_epochs=n_epochs, n_neurons=n_neurons)
    y_pred_train = lstm_model.predict(train_X, batch_size=batch_size, verbose=1)
    rmse_train = mean_squared_error(train_y, y_pred_train)
    print ('Train MSE: %.3f' % rmse_train)
    # make batch prediction
    y_pred = lstm_model.predict(test_X, batch_size=batch_size, verbose=1)
    #for i in range(test_y.shape[0]):
        #print ('Excepted=%.3f, Predicted=%.3f' % (test_y[i], y_pred[i]))
    rmse =mean_squared_error(test_y, y_pred)
    print ('Test MSE: %.3f' % rmse)
    plt.plot(tra_loss, 'b', label='traing_loss')
    plt.plot(val_loss, 'g', label='validation_loss')
    plt.legend(loc='lower right')
    plt.show()

    plt.title('testset results')
    plt.plot(test_y, 'b', label='ground truth')
    plt.plot(y_pred, 'r', label='prediction')
    plt.legend(loc='lower right')
    plt.show()

    plt.title('training set results')
    plt.plot(train_y[0:1000], 'b', label='ground truth')
    plt.plot(y_pred_train[0:1000], 'r', label='prediction')
    plt.legend(loc='lower right')
    plt.show()
