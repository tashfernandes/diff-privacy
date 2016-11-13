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

        self.data = np.asarray(data)
        self.target = np.asarray(target)

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target
