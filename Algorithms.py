import sklearn as sk
import numpy as np
from sklearn.linear_model import LogisticRegression

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

    def print_report(self):
        print(sk.metrics.classification_report(self.expected, self.predicted))
        print(sk.metrics.confusion_matrix(self.expected, self.predicted))

    def model(self):
        return self._model

    def accuracy(self):
        return float(np.sum(self.expected == self.predicted)) / float(len(self.expected))

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

        print "Noise vec: ", noise_vecs.shape
        print "Model: ", model_coef.shape

        # Let's check we've done the calc right...
        noise_dim = noise_vecs.shape[0]
        model_dim = model_coef.shape[1]
        #if noise_dim != model_dim:
        #    raise Exception("Dimensions for noise vector are incorrect, should be ", model_dim)

        noisy_model = np.add(model_coef, noise_vecs)
        #print "Old model: ", model_coef
        #print "New model: ", noisy_model
        self._model.coef_ = noisy_model

        self.predicted = self._model.predict(data)
        self.expected = target

    def print_report(self):
        print(sk.metrics.classification_report(self.expected, self.predicted))
        print(sk.metrics.confusion_matrix(self.expected, self.predicted))

    def accuracy(self):
        return float(np.sum(self.expected == self.predicted)) / float(len(self.expected))

