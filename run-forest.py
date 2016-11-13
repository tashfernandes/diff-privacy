#!/usr/local/bin/python

from DPData import CsvData
from DPData import ForestData
from ML import MLData
from Algorithms import Regression
from Algorithms import DPRegression
from Privacy import Privacy

dataset = ForestData( CsvData('../forestdata/raw/tx_regression.csv') )
mldata = MLData(dataset, split=0.5)

regr = Regression( { 'lambda' : 0.2 } )
training = mldata.training()
regr.train(training.data(), training.target())
test = mldata.test()
regr.test(test.data(), test.target())
regr.print_report()

# Now let's add differential privacy to the model

training_data =  training.data()
epsilon = 0.1
privacy = Privacy(lda=0.2, data_size=len(training_data), dim=training_data.shape[1], epsilon=epsilon)
private_regr = DPRegression( regr, privacy )
