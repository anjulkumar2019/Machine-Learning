from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MVC(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weight=None):
        self.classifiers = classifiers
        self.n_cl = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weight = weight
        
    def fit(self, X, y):
        self.le = LabelEncoder()
        self.le.fit(y)
        self.classes = self.le.classes_
        self.model_classifier = []
        for classif in self.classifiers:
            fitted_classif = clone(classif).fit(X,self.le.transform(y))
            self.model_classifier.append(fitted_classif)
        return(self)

    def predict(self, X):
        if self.vote == 'probability':
            majority_vote = np.argmax(self.predict_proba(X),axis=1)
        else: 
            predictions = np.asarray([classif.predict(X) for classif in self.model_classifier]).T
            majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weight=self.weight)),axis=1,arr=predictions)
            majority_vote = self.le.inverse_transform(majority_vote)
            return(majority_vote)
        
    def predict_proba(self, X):
        prob = np.asarray([classif.predict_proba(X) for classif in self.model_classifier])
        average_proba = np.average(prob, axis=0, weight=self.weight)
        return(average_proba)
