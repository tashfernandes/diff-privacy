class MLDataSet:

    def __init__(self, data, target):
        self._data = data
        self._target = target

    def data(self):
        return self._data

    def target(self):
        return self._target

class MLData:
    '''This class holds data for use in machine learning.'''

    def __init__(self, data, split=None):
        # Split into training and test
        self._training, self._test = self._split(data, split)

    def _split(self, data, percent):
        '''Split the data into training and test sets. Assumes the dataset is already randomised.'''

        full_data = data.get_data()
        full_target = data.get_target()
        if percent == None:
            return MLDataSet(full_data, full_target), MLDataSet(full_data, full_target)

        train_length = int(percent * len(full_data))
        training_data = full_data[0:train_length]
        training_target = full_target[0:train_length]
        test_data = full_data[train_length:]
        test_target = full_target[train_length:]
        return MLDataSet(training_data, training_target), MLDataSet(test_data, test_target)

    def training(self):
        return self._training

    def test(self):
        return self._test

class MLAlgorithm:
    '''Holds the algorithm used for machine learning.'''

    def __init__(self, alg):
        self.alg = alg

    def run(self, data):
        '''Run the given algorithm on the dataset.'''
        raise Exception("Unimplemented method")

class MLRunner:
    '''This class executes an ML algorithm on some dataset.'''
    
    def __init__(self, algorithm, data):
        self.alg = algorithm
        self.data = data

    def execute(self):
        self.alg.run(self.data)
