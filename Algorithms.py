import sklearn as sk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

class Report:

    def __init__(self, expected, predicted):
        self._expected = expected
        self._predicted = predicted

    def print_report(self):
        print(sk.metrics.classification_report(self._expected, self._predicted))
        print(sk.metrics.confusion_matrix(self._expected, self._predicted))

    def accuracy(self):
        #return float(np.sum(self._expected == self._predicted)) / float(len(self._expected))
        return sk.metrics.accuracy_score(self._expected, self._predicted)

class Regression:
    
    def __init__(self, params):
        self._params = params
        self._model = LogisticRegression(
            C=1.0/self._params['lambda'],
            solver='lbfgs',
            multi_class='multinomial')

    def train(self, data, target):
        self._model.fit(data, target)

    def test(self, data, target):
        self.predicted = self._model.predict(data)
        self.expected = target
        self._report = Report(self.expected, self.predicted)
        return self._report

    def model(self):
        return self._model

    def params(self):
        return self._params


class DPRegression:

    # This class assumes that the regression model passed in has
    # already had training data run over it.
    def __init__(self, model, privacy):
        self._model = model
        self._privacy = privacy

    def test(self, data, target):
        '''Adds differential privacy to the model and then runs the prediction.'''
        # First save the coefficients so we can replace them afterwards!!!

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
        self._model.coef_ = model_coef

        self.expected = target


        self._report = Report(self.expected, self.predicted)
        return self._report


class NaiveBayes:

    def __init__(self, params):
        self.params = params
        if not 'alpha' in self.params:
            self.params['alpha'] = 1.0
        self._model = MultinomialNB(alpha=self.params['alpha'])
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

    def train(self, data, target):
        X_train = self._vectorizer.fit_transform(data)
        self._model.fit(X_train, target)

    def test(self, data, target):
        X_test = self._vectorizer.transform(data)
        self.predicted = self._model.predict(X_test)
        self.expected = target
        self._report = Report(self.expected, self.predicted)
        return self._report

    def model(self):
        return self._model


class DPNaiveBayes:

    def __init__(self, model):
        self._model = model
