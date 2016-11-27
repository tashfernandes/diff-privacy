from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv

class CsvData:
    
    def __init__(self, filename):
        '''Loads the data in the given file and returns a data array.'''
        reader = None
        try:
            f = open(filename, 'r')
            self.reader = csv.DictReader(f)

        except csv.Error as e:
            print "Error opening csv: ", e

    def __iter__(self):
        return self

    def next(self):
        try:
            return self.reader.next()
        except:
            raise StopIteration

class ForestData:

    def __init__(self, csvdata):
        data = []
        target = []
        for d in csvdata:
            y = int(d['Cover_Type'])
            del d['Cover_Type']
            vals = d.values()
            # let's get rid of the id field
            vals = vals[1:]
            vals_converted = [float(x) for x in vals]
            data.append(vals_converted)
            target.append(y)

        # randomise the data
        rand_array = np.random.permutation(len(data))
        d = np.array(data)
        self._data = d[rand_array]
        t = np.array(target)
        self._target = t[rand_array]

    def data(self):
        return self._data

    def target(self):
        return self._target

class NewsgroupDataSet:
       
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data.data

    def target(self):
        return self._data.target


class NewsgroupsData:

    def __init__(self, categories):
        self._training = NewsgroupDataSet( fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories) )
        self._test = NewsgroupDataSet(fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories))

    def training(self):
        return self._training
    
    def test(self):
        return self._test

class VectorDataSet:

    def __init__(self, data, target):
        self._data = data
        self._target = target

    def data(self):
        return self._data

    def target(self):
        return self._target

class VectorData:

    def __init__(self, datamodel):
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        training = datamodel.training()
        train_data = self._vectorizer.fit_transform(training.data())
        train_target = training.target()
        self._training = VectorDataSet(train_data, train_target)

        test = datamodel.test()
        test_data = self._vectorizer.transform(test.data())
        test_target = test.target()
        self._test = VectorDataSet(test_data, test_target)

    def training(self):
        return self._training

    def test(self):
        return self._test
