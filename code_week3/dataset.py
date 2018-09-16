import numpy as np
import math
import matplotlib.pyplot as plt


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


def dataset_add_anomaly(x, y, anomaly_interval, std):
    x_a = np.array(np.arange(0., len(x), anomaly_interval))

    anomalies = []
    y_with_a = np.copy(y)

    for i in range(len(x_a)):
        p = np.random.random()
        if p > 0.5:
            anomaly = y[int(x_a[i])] + (1.3 + p) * std
        else:
            anomaly = y[int(x_a[i])] - (1.3 + p) * std
        anomalies.append(anomaly)

        y_with_a[int(x_a[i])] = anomaly

    return x_a, anomalies, y_with_a


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
    time_range = 500
    std = 3
    with_noise = False
    anomaly_interval = 40
    period = 200

    d1 = dataset_selection(5, time_range,with_noise,period,anomaly_interval,std,False)
    x, y = d1["x"], d1["y"]
    visualize_ci(x, y, std)

    d2 = dataset_selection(6, time_range,with_noise,period,anomaly_interval,std,False)
    x, y, x_a, y_org, anomalies = d2["x"], d2["y"], d2["x_a"], d2["y_org"], d2["a"]
    visualize_ci_anomaly(x, y, std, x_a, anomalies)
