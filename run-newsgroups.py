#!/usr/local/bin/python

from DPData import NewsgroupsData
from DPData import VectorData
from ML import MLData
from Algorithms import Regression, DPRegression
from Privacy import DiffPrivacy

import csv

categories = None

# For this data we run using differing sensitivities

dataset = VectorData(NewsgroupsData(categories=categories))

mldata = MLData(dataset.training(), dataset.test())
training = mldata.training()
test = mldata.test()

lda = 10.0
params = {'lambda': lda}
regr = Regression(params)
regr.train(training.data(), training.target())
report = regr.test(test.data(), test.target())

acc = report.accuracy()
print "Got accuracy: ", acc

results = []
training_data = training.data()

# Now let's add differential privacy to the model
for epsilon in (10.0, 8.0, 5.0, 3.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0001):
    privacy_params = {'lambda' : lda, 'data_size' : training_data.shape[0], 'dim' : training_data.shape[1]}
    privacy = DiffPrivacy(epsilon=epsilon, params=privacy_params)
    private_regr = DPRegression( regr.model(), regr_params )
    dp_report = private_regr.test(test.data(), test.target())

    dp_acc = dp_report.accuracy()

    results.append( { 'size' : training_data.shape[0], 'epsilon': epsilon, 'lambda' : lda, 'accuracy' : acc, 'dp_accuracy' : dp_acc } )

if len(results) > 0:
    filename = 'ng_results.csv'
    output_file = open(filename, 'wb')
    writer = csv.DictWriter(output_file, fieldnames = results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    output_file.close()

