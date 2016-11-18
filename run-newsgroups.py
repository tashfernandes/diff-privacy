#!/usr/local/bin/python

from DPData import NewsgroupsData
from ML import MLData
from Algorithms import NaiveBayes
from Privacy import Privacy

import csv

categories = None

dataset = NewsgroupsData(categories=categories)

mldata = MLData(dataset.training(), dataset.test())
training = mldata.training()
test = mldata.test()

model = NaiveBayes({'alpha' : 0.01})
model.train(training.data(), training.target())
report = model.test(test.data(), test.target())

acc = report.accuracy()
print "Got accuracy: ", acc

