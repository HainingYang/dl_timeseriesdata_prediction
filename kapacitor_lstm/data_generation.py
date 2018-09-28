#!/usr/bin/python2

import numpy as np
from numpy import random
from datetime import timedelta, datetime
import sys
import time
import requests
import math


write_url = 'http://localhost:9092/write?db=lstmData&rp=autogen&precision=s'
measurement = 'sale'

def dataset_with_trend_and_season(time_range, with_noise, period):
    x = np.array(np.arange(0, time_range))
    y = 10*(np.sin(np.pi * (x-25) / 50) + np.cos(np.pi * (x-25) / 25))+20
    noise = np.random.uniform(-0.2, 0.2, time_range)
    if with_noise:
        y = y + noise

    for i in range(0, time_range):
        y[i] += 0.3 * (i-math.floor(i/period)*period)

    return x, y


def main():
    total_time_range = 6004  # dataset contains (total_time_range-seq_l) many data samples
    with_noise = False
    period = 400
    x, y = dataset_with_trend_and_season(total_time_range, with_noise, period)

    now = datetime(2018, 9, 24)
    second = timedelta(seconds=1)
    epoch = datetime(2018, 9, 23)

    points = []
    for i in range(len(y)):
        points.append('%s value=%d %d' %(
            measurement,
            y[i],
            (now-epoch).total_seconds(),
        ))
        now += second

    # Write data to Kapacitor
    r = requests.post(write_url, data='\n'.join(points))
    if r.status_code != 204:
        print >> sys.stderr, r.text
        return 1
    return 0

if __name__ == '__main__':
    exit(main())