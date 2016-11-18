class MLDataSet:

    def __init__(self, data, target):
        self._data = data
        self._target = target

    def data(self):
        return self._data

    def target(self):
        return self._target

class MLDataGenerator:

    ## Class method for creating datasets for use in MLData
    # Subset holds the number of elements to select for the dataset
    # Split holds the fraction of elements in the full dataset to use in training
    # Data class has methods data() and target() to return the full dataset and target data    

    @staticmethod
    def createTrainTestData(data, subset=None, split=None):

        full_data = data.data()
        full_target = data.target()

        if subset != None:
            full_data = full_data[0:subset]
            full_target = full_target[0:subset]

        if split == None:
            return MLDataSet(full_data, full_target), MLDataSet(full_data, full_target)

        train_length = int(split * len(full_data))
        training_data = full_data[0:train_length]
        training_target = full_target[0:train_length]
        test_data = full_data[train_length:]
        test_target = full_target[train_length:]
        return MLDataSet(training_data, training_target), MLDataSet(test_data, test_target)

class MLData:
    '''This class holds data for use in machine learning.'''
    # train_dataset and test_dataset have type which contains data and target
    def __init__(self, train_dataset, test_dataset):
        self._training = MLDataSet(train_dataset.data(), train_dataset.target())
        self._test = MLDataSet(test_dataset.data(), test_dataset.target())

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
