# diff-privacy

Source code implementing differential privacy on logistic regression as
described in Chaudhuri et al.'s paper "Privacy-Preserving Logistic Regression".

This code requires Python 2.7.

* parameter_estimation.py - code for estimating regularisation parameter for 20 newsgroups dataset
* run-newsgroups.py - runs logistic regression on the 20 newsgroups dataset as loaded from Scikit-Learn.
* run-forest.py - runs logistic regression on the Forest Cover Type dataset. This needs to be downloaded
                  from https://kdd.ics.uci.edu/databases/covertype/covertype.data.html
