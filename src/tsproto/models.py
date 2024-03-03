import warnings
import pandas as pd
import numpy as np
import ruptures as rpt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.tree import DecisionTreeClassifier
import time
from tslearn.clustering import KShape
from kshape.core import KShapeClusteringCPU
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans
from numpy import mean
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

from tsproto.utils import dominant_frequencies_for_rows

"""
Documentation of this module

"""


def foo():
    """
    Just a sample function to see if the docs fail on this file or on classes
    :return:
    """
    pass





class InterpretableModel:
    def fit_or_predict(ohe_train, features, target, intclf=None, verbose=0, max_depth=None, min_samples_leaf=0.05,
                       weights=None):
        if intclf is None:
            intclf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            intclf.fit(ohe_train[features], ohe_train[target], sample_weight=weights)
        iclf_pred = intclf.predict(ohe_train[features])
        if verbose > 0:
            print(f'Tree depth: {intclf.get_depth()}')
        if ohe_train[target].nunique() > 2:
            average = 'macro'
        else:
            average = 'binary'
        return (accuracy_score(ohe_train[target], iclf_pred),
                precision_score(ohe_train[target], iclf_pred, average=average),
                recall_score(ohe_train[target], iclf_pred, average=average),
                f1_score(ohe_train[target], iclf_pred, average=average),
                intclf)
