from kapacitor.udf.agent import Agent, Handler
from kapacitor.udf import udf_pb2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
                    filename='lstmTrain.log', filemode='w')
logger = logging.getLogger()

class OutliersTrainer(Handler):
    class LSTMTrainer(object):
        def __init__(self):
            self._neurons = 7
            self._batch = 100
            self._epoch = 700
            self._features = 1
            self._seq = 0
            self._model = None
            self._dataset = []

        def build_network(self, seq):
            logger.info('build lstm started')
            self._seq = seq
            self._model = Sequential()
            #logger.info('finish adding sequence')
            self._model.add(LSTM(self._neurons, batch_input_shape=(self._batch,seq, self._features),stateful=True))
            #logger.info('finish adding lstm layer')
            self._model.add(Dense(units=1))
            #logger.info('finish adding dense layer')
            self._model.compile(loss='mean_squared_error', optimizer='adam')
            logger.info('build lstm finished')

        def reset(self):
            self._dataset = []

        def update(self, value):
            self._dataset.append(value)

        def train_und_export(self, path):
            tra_loss = []
            X, y = self.data_preparation()
            logger.info('start training')
            for i in range(self._epoch):
                history = self._model.fit(X, y, epochs=1, batch_size=self._batch, verbose=1,shuffle=False)
                tra_loss.append(history.history['loss'])
                logger.info("loss in epoche " + str(i) + " is: " + str(tra_loss[i]))
                self._model.reset_states()

            logger.info('end of training')
            logger.info ('start exporting')
            self._model.save(path)
            logger.info('end of exporting')
            return True

        def data_preparation(self):
            logger.info('from lstm perspective '+ str(len(self._dataset)) + ' points are stored in dataset')
            data = np.reshape(self._dataset, (len(self._dataset), self._features))
            n_samples = data.shape[0]
            X = list()
            y = list()
            # create input sequences for lstm w.r.t. the given sequence length
            for i in range(0, n_samples - self._seq, 1):
                sample = data[i:i + self._seq]
                X.append(sample)
                y.append(data[i + self._seq])

            # convert input into a 2D array
            X = np.array(X)  # suitable dimension for the lstm input
            y = np.array(y)

            return X, y

    def __init__(self, agent):
        self._agent= agent
        self._field= '' # define which column in the database to read
        self._sequence = 0 # the length of window
        self._path = '' # place where the trained model be stored

        self._trainer = OutliersTrainer.LSTMTrainer()
        self._begin_response = None
        self._count = 0
        logger.info('OutliersTrainer initiated')

    def info(self):
        logger.info('info started')
        response = udf_pb2.Response()

        # Define the input and output data edge
        response.info.wants = udf_pb2.BATCH
        response.info.provides = udf_pb2.STREAM

        # Define the values that passed from tick script
        response.info.options['field'].valueTypes.append(udf_pb2.STRING)
        response.info.options['sequence'].valueTypes.append(udf_pb2.INT)
        response.info.options['path'].valueTypes.append(udf_pb2.STRING)

        logger.info("info finished")
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
            elif opt.name == 'path':
                self._path = opt.values[0].stringValue

        if self._sequence <= 0:
            success = False
            msg += ' must supply window size > 0'
        if self._field == '':
            success = False
            msg += ' must supply a field name'
        if self._path == '':
            success =False
            msg += 'must supply a path name'

        response = udf_pb2.Response()
        response.init.success = success
        response.init.error = msg[1:]

        # build the network
        self._trainer.build_network(self._sequence)
        logger.info('init finished')
        return response

    def begin_batch(self, begin_req):
        # Keep copy of begin_batch
        logger.info('begin batch started')
        self._trainer.reset()
        logger.info('begin batch finished')

    def point(self, point):
        self._count += 1
        value = point.fieldsDouble[self._field]
        # pass the data to lstm model
        self._trainer.update(value)

    def end_batch(self, end_req):
        logger.info('from batch perspective ' + str(self._count)+ ' points are collected')
        # after loading all data points in the batch, start training
        result = self._trainer.train_und_export(self._path)

        response = udf_pb2.Response()
        response.end.CopyFrom(end_req)
        end_meta = response.end
        response = udf_pb2.Response()
        response.point.time = end_meta.tmax
        response.point.name = end_meta.name
        response.point.group = end_meta.group
        response.point.tags.update(end_meta.tags)
        response.point.fieldsBool["result"] = result
        self._agent.write_response(response)
        logger.info('finish end_batch')

    def snapshot(self):
        response = udf_pb2.Response()
        response.snapshot.snapshot = ''

        return response

    def restore(self, restore_req):
        response = udf_pb2.Response()
        response.restore.success = False
        response.restore.error = 'not implemented'

        return response

if __name__ == '__main__':
    a = Agent()
    h = OutliersTrainer(a)
    a.handler = h

    logger.info("Starting Agent")
    a.start()
    a.wait()
    logger.info("Agent finished")
