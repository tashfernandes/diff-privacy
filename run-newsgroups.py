#!/usr/local/bin/python

from DPData import NewsgroupsData
from DPData import VectorData
from ML import MLData
from Algorithms import Regression, DPRegression
from Privacy import DiffPrivacy

import numpy as np
import csv

categories = None

dataset = VectorData(NewsgroupsData(categories=categories))

mldata = MLData(dataset.training(), dataset.test())
training = mldata.training()
test = mldata.test()
training_data = training.data()
arr = np.arange(training_data.shape[0])

for lda in (0.001,0.01,0.1,1.0,5.0,10.0):
    params = {'lambda': lda}
    regr = Regression(params)
    regr.train(training.data(), training.target())
    report = regr.test(test.data(), test.target())
    
    acc = report.accuracy()
    print "Got accuracy: ", acc
    
    results = []
    
    # Let's make only a subset of the dimensions sensitive
    # shuffle numbers from 0 - dim and pick first subdim
    for subdim in (50000,20000,10000,1000,100):
        for run in range(10):
            np.random.shuffle(arr)
            indices = arr[:subdim]
        
            # Now let's add differential privacy to the model
            for epsilon in (10.0, 8.0, 5.0, 3.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0001):
                privacy_params = {'lambda' : lda, 'data_size' : training_data.shape[0], 'dim' : training_data.shape[1], 'subdim':indices}
                privacy = DiffPrivacy(epsilon=epsilon, params=privacy_params)
                private_regr = DPRegression( regr.model(), privacy )
                dp_report = private_regr.test(test.data(), test.target())
            
                dp_acc = dp_report.accuracy()
            
                results.append( { 'run' : run, 'dim' : subdim, 'epsilon': epsilon, 'lambda' : lda, 'accuracy' : acc, 'dp_accuracy' : dp_acc } )

    # Now do a run with the entire feature space just for double checking awesomeness
    for epsilon in (10.0, 8.0, 5.0, 3.0, 1.0, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0001):
        privacy_params = {'lambda' : lda, 'data_size' : training_data.shape[0], 'dim' : training_data.shape[1]}
        privacy = DiffPrivacy(epsilon=epsilon, params=privacy_params)
        private_regr = DPRegression( regr.model(), privacy )
        dp_report = private_regr.test(test.data(), test.target())
            
        dp_acc = dp_report.accuracy()
    
        results.append( { 'run' : 1, 'dim' : training_data.shape[1], 'epsilon': epsilon, 'lambda' : lda, 'accuracy' : acc, 'dp_accuracy' : dp_acc } )
    
    if len(results) > 0:
        filename = 'subdim_results_' + str(lda) + '.csv'
        output_file = open(filename, 'wb')
        writer = csv.DictWriter(output_file, fieldnames = results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)
        output_file.close()

