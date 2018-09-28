from kapacitor.udf.agent import Agent, Handler
from kapacitor.udf import udf_pb2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import numpy as np
from sklearn.metrics import mean_squared_error
import math

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
                    filename='/tmp/lstmTest.log', filemode='w')
logger = logging.getLogger()


class OutliersHandler(Handler):
    class LSTMPredictor(object):
        def __init__(self):
            self._neurons = 7
            self._batch = 1
            self._dataset = []
            self._features = 1
            self._model = None
            self._std = 0.0
            self._seq = 0

        def reset(self):
            self._dataset = []

        def update(self, value, point):
            self._dataset.append((value, point))

        def load_model(self, path, seq, std):
            logger.info('build lstm finished')
            logger.info('start reloading lstm model')
            model = load_model(path)
            self._seq = seq
            self._std = std
            # modify batch size for point-wise prediction
            logger.info('rebuild lstm started')
            self._seq = seq
            self._model = Sequential()
            self._model.add(LSTM(self._neurons, batch_input_shape=(self._batch, seq, self._features), stateful=True))
            self._model.add(Dense(units=1))
            # copy weights
            weights = model.get_weights()
            self._model.set_weights(weights)
            self._model.compile(loss='mean_squared_error', optimizer='adam')

            logger.info('after reloading lstm model')

        def predict(self):
            logger.info('begin predict')
            anomalies = []
            data = []
            for i in range(len(self._dataset)):
                data.append(self._dataset[i][0])
            X, y = self.data_preparation(data)
            y_pred = self._model.predict(X, verbose=1)
            rmse = math.sqrt(mean_squared_error(y, y_pred))
            logger.info('error of prediction: '+ str(rmse))
            #for i in range(len(y)):
            #    if abs(y[i]-y_pred[i]) > self._std:
            #        anomalies.append(self._dataset[i][1])  # add point in result
            logger.info('finish prediction')
            return y_pred

        def data_preparation(self, dataset):
            logger.info('begin data preparation')
            logger.info('from lstm perspective ' + str(len(dataset)) + ' points are stored in dataset')
            data = np.reshape(dataset, (len(dataset), self._features))
            n_samples = data.shape[0]
            X = list()
            y = list()

            for i in range(0, n_samples - self._seq, 1):
                sample = data[i:i + self._seq]
                X.append(sample)
                y.append(data[i + self._seq])

            # convert input into a 2D array
            X = np.array(X)  # suitable dimension for the lstm input
            y = np.array(y)
            logger.info('end data preparation')
            return X, y

    def __init__(self, agent):
        self._agent = agent

        self._field = ''
        self._sequence = 0
        self._std = 0.0
        self._path = ''

        self._predictor = OutliersHandler.LSTMPredictor()
        self._begin_response = None
        self._count = 0
        self._points=[]
        logger.info('OutliersHandler initiated')

    def info(self):
        """
        Respond with which type of edges we want/provide and any options we have.
        """
        logger.info('info started')
        response = udf_pb2.Response()

        # Define the input output data edge
        response.info.wants = udf_pb2.BATCH
        response.info.provides = udf_pb2.STREAM

        # Define which field to process and value passed from tickscript
        response.info.options['field'].valueTypes.append(udf_pb2.STRING)
        response.info.options['sequence'].valueTypes.append(udf_pb2.INT)
        response.info.options['std'].valueTypes.append(udf_pb2.DOUBLE)
        response.info.options['path'].valueTypes.append(udf_pb2.STRING)

        logger.info('info finished')
        return response

    def init(self, init_req):
        """
            Given a list of options initialize this instance of the handler
        """
        logger.info('init started')
        success = True
        msg = ''

        for opt in init_req.options:
            if opt.name == 'field':
                self._field = opt.values[0].stringValue
            if opt.name == 'sequence':
                self._sequence = opt.values[0].intValue
            if opt.name == 'path':
                self._path = opt.values[0].stringValue
            elif opt.name == 'std':
                self._std = opt.values[0].doubleValue

        if self._sequence <= 0:
            success = False
            msg += ' must supply window size > 0'
        if self._field == '':
            success = False
            msg += ' must supply a field name'
        if self._std < 0:
            success = False
            msg += 'must supply a std value >0 '
        if self._path == '':
            success = False
            msg += ' must supply a field name'

        self._predictor.load_model(self._path, self._sequence, self._std)
        response = udf_pb2.Response()
        response.init.success = success
        response.init.error = msg[1:]

        logger.info('init finished')
        return response

    def begin_batch(self, begin_req):
        logger.info('start begin_batch')
        self._predictor.reset()
        logger.info('end begin_batch')

    def point(self, point):
        self._count += 1
        self._points.append(point)
        value = point.fieldsDouble[self._field]
        self._predictor.update(value, point)

    def end_batch(self, batch_meta):
        logger.info('from batch perspective ' + str(self._count) + ' points are collected')
        logger.info('begin end_batch')
        prediction = self._predictor.predict() # single point
        logger.info(type(prediction))
        logger.info('prediction ' + str(prediction))
  
        response = udf_pb2.Response()
        response.point.time = batch_meta.tmax
        response.point.name = batch_meta.name
        response.point.group = batch_meta.group
        response.point.tags.update(batch_meta.tags)

        response.point.fieldsDouble["value"]=prediction
        self._agent.write_response(response)

        logger.info('finish end_batch')

if __name__ == '__main__':
    a = Agent()
    h = OutliersHandler(a)
    a.handler = h

    logger.info("Starting Agent")
    a.start()
    a.wait()
    logger.info("Agent finished")
