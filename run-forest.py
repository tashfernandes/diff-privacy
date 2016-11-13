#!/usr/local/bin/python

from DPData import CsvData
from DPData import ForestData
from ML import MLData
from Algorithms import Regression
from Algorithms import DPRegression
from Privacy import Privacy

dataset = ForestData( CsvData('../forestdata/raw/tx_regression.csv') )
#dataset = ForestData( CsvData('sample.csv') )
mldata = MLData(dataset, split=0.8)

# TODO: Check that this lambda is actually being used

for i, C in enumerate((10.0,2.0,1.0,0.1,0.05,0.01,0.001)):
    regr = Regression( { 'lambda' : C } )
    training = mldata.training()
    regr.train(training.data(), training.target())
    test = mldata.test()
    regr.test(test.data(), test.target())
    regr.print_report()

    acc = regr.accuracy()
    print "Accuracy: ", acc

    # Now let's add differential privacy to the model

    training_data =  training.data()
    epsilon = 0.1
    privacy = Privacy(lda=C, data_size=len(training_data), dim=training_data.shape[1], epsilon=epsilon)
    private_regr = DPRegression( regr.model(), privacy )
    private_regr.test(test.data(), test.target())
    private_regr.print_report()

    acc = private_regr.accuracy()
    print "Accuracy: ", acc
