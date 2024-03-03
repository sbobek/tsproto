import warnings
import ruptures as rpt
#from tslearn.clustering import TimeSeriesKMeans
#from tslearn.utils import to_time_series_dataset
from sklearn.tree import DecisionTreeClassifier
import time
from tslearn.clustering import KShape
from kshape.core import KShapeClusteringCPU
#from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans
from numpy import mean
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin









"""
Documentation of this module

"""


def foo():
    """
    Just a sample function to see if the docs fail on this file or on classes
    :return:
    """
    pass

