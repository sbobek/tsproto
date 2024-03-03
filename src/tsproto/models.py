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

