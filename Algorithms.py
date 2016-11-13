import sklearn as sk
from sklearn.linear_model import LogisticRegression

class Regression:
    
    def __init__(self, params):
        self.params = params

    def train(self, data, target):
        self.model = LogisticRegression(
            C=self.params['lambda'],
            solver='lbfgs',
            multi_class='multinomial')
        
        self.model.fit(data, target)

    def test(self, data, target):
        self.predicted = self.model.predict(data)
        self.expected = target

    def print_report(self):
        print(sk.metrics.classification_report(self.expected, self.predicted))
        print(sk.metrics.confusion_matrix(self.expected, self.predicted))

    def coef(self):
        return self.model.coef_

class DPRegression:

    def __init__(self, regression, privacy):
        self.regression = regression
        self.privacy = privacy

    def train(self, data, target):
        self.regression.train(data, target)
        # We need to add noise to the vector now
        model_coef = self.regression.coef()
        # model_coef is a numpy ndarray containing one vector for each classification
        noise_vec = privacy.generate_noise_vector()  # TODO
    

