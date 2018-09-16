import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import dataset


def data_preparation(data, win_l):
    # exact number of samples and its feature dimension
    n_samples, n_features = data.shape[0], data.shape[1]
    # create input sequences for lstm w.r.t. the given sequence length
    X = list()
    y = list()
    for i in range(0, n_samples-win_l, 1):
        sample = data[i:i+win_l]
        X.append(sample)
        y.append(data[i+win_l])

    # convert input into a 3D array
    X = np.array(X)
    y = np.array(y)

    return X, y


def fit_mlp(X, y, test_X, test_y, batch_size, n_epoches, n_neurons):
    model = Sequential()
    model.add(Dense(units=n_neurons[0], input_dim=X.shape[1], activation='relu'))
    for i in range(1, len(n_neurons)-1):
        model.add(Dense(units=n_neurons[i], activation='relu'))
    model.add(Dense(units=n_neurons[-1]))

    model.compile(loss='mean_squared_error', optimizer='adam')
    history= model.fit(X,y,epochs=n_epoches, batch_size=batch_size, verbose=2, validation_data=(test_X,test_y))

    return model, history


if __name__=='__main__':
    total_time_range = 10004
    with_noise = False
    anomaly_interval = 200
    std = 3

    batch_size = 100
    test_size = 1000
    window_length = 7

    n_epochs = 500
    n_neurons = [15,5,1]

    x, y = dataset.normal_dataset_generator(total_time_range, with_noise)
    x_a, anomalies, y_with_a = dataset.dataset_add_anomaly(x, y, anomaly_interval, std)
    y_with_a = np.reshape(y_with_a, (total_time_range, 1))
    # dataset.visualize_ci_anomaly(x[:500],y[:500],std, x_a[:2], anomalies[:2])

    d_X, d_y = data_preparation(y_with_a, window_length)
    d_X = np.squeeze(d_X)
    d_y = np.squeeze(d_y)
    train_X, train_y = d_X[0:-test_size], d_y[0:-test_size]  # 1900 groups for training
    test_X, test_y = d_X[-test_size:], d_y[-test_size:]  # 100 groups for testing

    fit_model, history = fit_mlp(train_X, train_y, test_X, test_y, batch_size, n_epochs, n_neurons)
    # estimate performance
    train_mse = fit_model.evaluate(train_X, train_y, verbose=0)
    print ('Train mse: %.3f' % train_mse)
    test_mse = fit_model.evaluate(test_X, test_y, verbose=0)
    print ('Test mse: %.3f' % test_mse)

    preds_train = fit_model.predict(train_X)
    preds_test = fit_model.predict(test_X)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(train_loss, 'g')
    plt.plot(val_loss, 'b')
    plt.legend(['loss', 'val_loss'],loc='upper right')
    plt.show()

    plt.title('test set results')
    plt.plot(test_y, 'b', label='ground truth')
    plt.plot(preds_test, 'r', label='prediction')
    plt.legend(loc='lower right')
    plt.show()

    plt.title('training set results')
    plt.plot(train_y[0:1000], 'b', label='ground truth')
    plt.plot(preds_train[0:1000], 'r', label='prediction')
    plt.legend(loc='lower right')
    plt.show()