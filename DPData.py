from sklearn.datasets import fetch_20newsgroups
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

        self._data = np.array(data)
        self._target = np.array(target)

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
        self._training = NewsgroupDataSet( fetch_20newsgroups(subset='train', categories=categories) )
        self._test = NewsgroupDataSet(fetch_20newsgroups(subset='test', categories=categories))

    def training(self):
        return self._training
    
    def test(self):
        return self._test

