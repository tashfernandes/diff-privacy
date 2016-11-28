# Parameter estimation for logistic regression
# on the 20 newsgroups dataset.
# Use 10-fold cross validation on the training set
# and compute the mean of the error rates.

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics


categories = None
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
feature_names = vectorizer.get_feature_names()
feature_names = np.asarray(feature_names)

print("Lambda,Error,Std")
# For each value of lambda we want to perform 10-fold CV and get
# average misclassification rate.
for lda in (0.0001, 0.0003, 0.0005, 0.0008, 0.001,0.003, 0.005, 0.008, 0.01,0.03, 0.05, 0.08, 0.1, 0.5, 1.0):
    C = 1.0/lda
    model = LogisticRegression(tol=1e-3, C=C, solver="lbfgs")
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print("%0.6f, %0.6f, %0.6f" % (lda, 1.0-scores.mean(), scores.std() * 2))
    #print("Lambda: %0.3f, Accuracy: %0.4f (+/- %0.3f)" % (lda, scores.mean(), scores.std() * 2))
    #model.fit(X_train, y_train)
    #pred = model.predict(X_test)
    #score = metrics.accuracy_score(y_test, pred)
    #print("Lambda: %0.3f, accuracy:   %0.3f" % (lda, score))
