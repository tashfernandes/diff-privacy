#!/usr/local/bin/python

from DPData import CsvData
from DPData import ForestData
from ML import MLDataGenerator
from Algorithms import Regression
from Algorithms import DPRegression
from Privacy import Privacy

import csv

dataset = ForestData( CsvData('../forestdata/raw/tx_regression.csv') )

results = []

for subset in (1000, 10000, 50000, 100000, 200000, 500000, None):
    training, test = MLDataGenerator.createTrainTestData(dataset, subset=subset, split=0.8)

    for C in (10.0,2.0,1.0,0.1,0.05,0.01,0.001):
        regr = Regression( { 'lambda' : C } )
        regr.train(training.data(), training.target())
        report = regr.test(test.data(), test.target())

        acc = report.accuracy()

        # Now let's add differential privacy to the model
        training_data = training.data()

        for epsilon in (1.0, 0.8, 0.5, 0.1, 0.08, 0.05, 0.01, 0.005, 0.001):
            privacy = Privacy(lda=C, data_size=len(training_data), dim=training_data.shape[1], epsilon=epsilon)
            private_regr = DPRegression( regr.model(), privacy )
            dp_report = private_regr.test(test.data(), test.target())

            dp_acc = dp_report.accuracy()

            results.append( { 'size' : len(training_data), 'epsilon': epsilon, 'lambda' : C, 'accuracy' : acc, 'dp_accuracy' : dp_acc } )


if len(results) > 0:
    output_file = open('results.csv', 'wb')
    writer = csv.DictWriter(output_file, fieldnames = results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    output_file.close()
