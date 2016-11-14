import sklearn as sk
import numpy as np
from sklearn.linear_model import LogisticRegression

class Report:

    def __init__(self, expected, predicted):
        self._expected = expected
        self._predicted = predicted

    def print_report(self):
        print(sk.metrics.classification_report(self._expected, self._predicted))
        print(sk.metrics.confusion_matrix(self._expected, self._predicted))

    def accuracy(self):
        return float(np.sum(self._expected == self._predicted)) / float(len(self._expected))

class Regression:
    
    def __init__(self, params):
        self.params = params

    def train(self, data, target):
        self._model = LogisticRegression(
            C=self.params['lambda'],
            solver='lbfgs',
            multi_class='multinomial')
        
        self._model.fit(data, target)

    def test(self, data, target):
        self.predicted = self._model.predict(data)
        self.expected = target
        self._report = Report(self.expected, self.predicted)
        return self._report

    def model(self):
        return self._model

class DPRegression:

    # This class assumes that the regression model passed in has
    # already had training data run over it.
    def __init__(self, model, privacy):
        self._model = model
        self._privacy = privacy

    def test(self, data, target):
        '''Adds differential privacy to the model and then runs the prediction.'''
        model_coef = self._model.coef_
        num_rows = model_coef.shape[0]
        noise_vecs = []
        for i in range(num_rows):
            noise_vecs.append(self._privacy.generate_noise_vector())
        noise_vecs = np.array(noise_vecs)

        # Let's check we've done the calc right...
        noise_dim = noise_vecs.shape[0]
        model_dim = model_coef.shape[1]
        #if noise_dim != model_dim:
        #    raise Exception("Dimensions for noise vector are incorrect, should be ", model_dim)

        noisy_model = np.add(model_coef, noise_vecs)
        self._model.coef_ = noisy_model

        self.predicted = self._model.predict(data)
        self.expected = target

        self._report = Report(self.expected, self.predicted)
        return self._report


