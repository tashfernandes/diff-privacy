#!/usr/local/bin/python

from DPData import CsvData
from DPData import ForestData
from ML import MLDataGenerator
from Algorithms import Regression, DPRegression
from Privacy import DiffPrivacy

import csv

for trial in range(10):
    dataset = ForestData( CsvData('../forestdata/raw/tx_regression.csv') )

    results = []

    for subset in (1000, 10000, 50000, 100000, 200000, 500000, None):
        training, test = MLDataGenerator.createTrainTestData(dataset, subset=subset, split=0.8)

        for lda in (10.0,2.0,1.0,0.1,0.05,0.01,0.001):
            regr = Regression( { 'lambda' : lda } )
            regr.train(training.data(), training.target())
            report = regr.test(test.data(), test.target())

            acc = report.accuracy()

            # Now let's add differential privacy to the model
            training_data = training.data()

            for epsilon in (10.0, 8.0, 5.0, 3.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0001):
                regr_params = {'lambda' : lda, 'data_size' : training_data.shape[0], 'dim' : training_data.shape[1]}
                privacy = DiffPrivacy(epsilon=epsilon, params=regr_params)
                private_regr = DPRegression( regr.model(), privacy )
                dp_report = private_regr.test(test.data(), test.target())

                dp_acc = dp_report.accuracy()

                results.append( { 'size' : training_data.shape[0], 'epsilon': epsilon, 'lambda' : lda, 'accuracy' : acc, 'dp_accuracy' : dp_acc } )


    if len(results) > 0:
        filename = 'results_' + str(trial) + '.csv'
        output_file = open(filename, 'wb')
        writer = csv.DictWriter(output_file, fieldnames = results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)
        output_file.close()
