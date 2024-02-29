import warnings

import pandas as pd
from itertools import cycle
from ruptures.utils import pairwise
import ruptures as rpt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.tree import DecisionTreeClassifier
import time
from tslearn.clustering import KShape
from kshape.core import KShapeClusteringCPU
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans
import importlib.util

from numpy import mean
from graphviz import Digraph
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

COLOR_CYCLE = ["#4286f4", "#f44174"]


def display(
        signal,
        true_chg_pts,
        computed_chg_pts=None,
        computed_chg_pts_color="k",
        computed_chg_pts_linewidth=3,
        computed_chg_pts_linestyle="--",
        computed_chg_pts_alpha=1.0,
        ax=None
):
    """Display a signal and the change points provided in alternating colors.
    If another set of change point is provided, they are displayed with dashed
    vertical dashed lines. The following matplotlib subplots options is set by
    default, but can be changed when calling `display`):

    - figure size `figsize`, defaults to `(10, 2 * n_features)`.

    Args:
        signal (array): signal array, shape (n_samples,) or (n_samples, n_features).
        true_chg_pts (list): list of change point indexes.
        computed_chg_pts (list, optional): list of change point indexes.
        computed_chg_pts_color (str, optional): color of the lines indicating
            the computed_chg_pts. Defaults to "k".
        computed_chg_pts_linewidth (int, optional): linewidth of the lines
            indicating the computed_chg_pts. Defaults to 3.
        computed_chg_pts_linestyle (str, optional): linestyle of the lines
            indicating the computed_chg_pts. Defaults to "--".
        computed_chg_pts_alpha (float, optional): alpha of the lines indicating
            the computed_chg_pts. Defaults to "1.0".
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.

    Returns:
        tuple: (figure, axarr) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """

    if type(signal) != np.ndarray:
        # Try to get array from Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    # let's set a sensible defaut size for the subplots
    # matplotlib_options = {
    #    "figsize": (10, 2 * n_features),  # figure size
    # }
    # add/update the options given by the user
    # matplotlib_options.update(kwargs)

    # create plots
    if ax is None:
        fig, ax = plt.subplots(sharex=True)

    # plot s
    ax.plot(range(n_samples), signal)
    color_cycle = cycle(COLOR_CYCLE)
    # color each (true) regime
    bkps = [0] + sorted(true_chg_pts)
    alpha = 0.2  # transparency of the colored background

    for (start, end), col in zip(pairwise(bkps), color_cycle):
        ax.axvspan(max(0, start - 0.5), end - 0.5, facecolor=col, alpha=alpha)
    # vertical lines to mark the computed_chg_pts
    if computed_chg_pts is not None:
        for bkp in computed_chg_pts:
            if bkp != 0 and bkp < n_samples:
                ax.axvline(
                    x=bkp - 0.5,
                    color=computed_chg_pts_color,
                    linewidth=computed_chg_pts_linewidth,
                    linestyle=computed_chg_pts_linestyle,
                    alpha=computed_chg_pts_alpha,
                )

    return ax


def outliers(data, multiplier=1.5):
    # finding the 1st quartile
    q1 = np.quantile(data, 0.25)

    # finding the 3rd quartile
    q3 = np.quantile(data, 0.75)

    # finding the iqr region
    iqr = q3 - q1

    # finding upper and lower whiskers
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    return (iqr, upper_bound, lower_bound)


class PrototypeEncoder():

    def __init__(self, blackbox, min_size, jump, pen, n_clusters, multiplier=1.5, method='kshape',
                 descriptors=['existance', 'duration', 'stats'], n_jobs=None, verbose=0, dims=1, sampling_rate=1,
                 feature_names=None):
        self.threshold = 0  # threshold for discarding slices of time-series of low importance
        self.multiplier = multiplier
        self.fixed_n_clusters = False
        self.n_clusters = {}
        self.jump = jump
        self.min_size = min_size
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pen = pen
        self.blackbox = blackbox
        self.descriptors = descriptors
        self.thresholds_ = [None] * dims
        self.kms_ = [None] * dims
        self.label_features_ = [None] * dims
        self.tsms_ = [None] * dims
        self.xbis_ = {}
        self.xbis_shap_ = {}
        self.xbis_cluster_labels_ = {}
        self.signal_ids_ = {}
        self.sampling_rate_ = sampling_rate
        self.weights_ = {}

        torch_spec = importlib.util.find_spec("torch")
        torch_found = torch_spec is not None
        if not torch_found and self.method == 'kshapegpu':
            self.method = 'kshape'
            warnings.warn("To use GPU version of KSHAPE, install TSProto with GPU support: pip install tsproto[gpu]")
        elif self.method == 'kshapegpu':
            from kshape.core_gpu import KShapeClusteringGPU

        if feature_names is None:
            self.feature_names = [str(f) for f in range(0, dims)]
        else:
            self.feature_names = feature_names

        if n_clusters is not None:
            self.fixed_n_clusters = True
            for dim in range(dims):
                self.n_clusters[self.feature_names[dim]] = n_clusters

    def fit(self, X, shapclass):
        return self._transform(X, shapclass, refit=True, transform=False)

    def transform(self, X, shapclass):
        return self._transform(X, shapclass, refit=False, transform=True)

    def fit_transform(self, X, shapclass):
        return self._transform(X, shapclass, refit=True, transform=True)

    def _transform(self, Xdim, shap_dim, refit=False, transform=True):
        train_dim = []
        features = []
        if self.verbose > 0:
            print(f'Dataset shape: {Xdim.shape}')
            start_time = time.time()
            print(f'Calculating changepoints...')

        for dim in range(0, Xdim.shape[2]):
            totalslice = []
            totalsslice = []
            indexes = []
            print(f'Dim: {dim}')
            X = Xdim[:, :, dim].reshape((Xdim.shape[0], Xdim.shape[1], 1))
            shapclass = shap_dim[:, :, dim].reshape((shap_dim.shape[0], shap_dim.shape[1], 1))
            mean_bkps = 0
            for i in range(0, X.shape[0]):
                algo = rpt.Pelt(model="rbf", min_size=self.min_size, jump=self.jump).fit(shapclass[i])
                breakpoints = algo.predict(pen=1)
                mean_bkps += len(breakpoints)
                # shapslices = [np.mean(abs(s)) for s in np.split(shapclass[i],breakpoints) if len(s) != 0]
                shapslices = [abs(s) for s in np.split(shapclass[i], breakpoints) if
                              len(s) != 0]  # (assuming this is abshap over all classses)
                slices = [s for s in np.split(X[i], breakpoints) if len(s) != 0]
                totalsslice.append(shapslices)
                totalslice.append(slices)
                indexes.append(np.ones(len(slices)) * i)

            print(len(totalsslice[0]))
            print(len(totalslice[0]))
            print(len(indexes[0]))

            ############ overriding clusters\
            if refit and (not self.fixed_n_clusters or self.fixed_n_clusters <= 1):
                self.n_clusters[self.feature_names[dim]] = max(2,
                                                               int(int(mean_bkps / X.shape[0]) * self.fixed_n_clusters))
                print(f'For {self.feature_names[dim]} c_clusters = {self.n_clusters[self.feature_names[dim]]}')

            ############# .
            if refit:
                if self.multiplier is not None:
                    concat_sslices = np.concatenate([item for sublist in totalsslice for item in sublist])
                    self.thresholds_[dim] = np.mean(concat_sslices) - self.multiplier * np.std(concat_sslices)
                else:
                    self.thresholds_[dim] = 0

            # ante-hoc filtering
            if self.thresholds_[dim] > 0:
                indexed_slices = [[i, s, ss] for ts, tss, ti in zip(totalslice, totalsslice, indexes) for s, ss, i in
                                  zip(ts, tss, ti) if ss > self.thresholds_[dim]]
                isdf = pd.DataFrame(indexed_slices)
                indexes = isdf.loc[:, 0]
                if len(np.unique(indexes)) < (np.max(indexes) - 1):
                    # in case there is a signal that has absolutely no readings
                    # reduce the threshold
                    print('WARNING: Changing the threshold, due to empty record')
                    isdf = pd.DataFrame(indexed_slices, columns=['index', 'slice', 'shapslice'])
                    self.thresholds_[dim] = isdf.groupby('index')['shapslice'].max().min()
                    indexed_slices = [[i, s, ss] for ts, tss, ti in zip(totalslice, totalsslice, indexes) for s, ss, i
                                      in zip(ts, tss, ti) if ss > self.thresholds_[dim]]
                    isdf = pd.DataFrame(indexed_slices)
                indexes = list(isdf.groupby(0)[0].apply(np.array))
                totalslice = list(isdf.groupby(0)[1].apply(list))
                totalsslice = list(isdf.groupby(0)[2].apply(list))

            ############# post-hoc filtering
            # TODO: filter whole clusters which average/max importance is below certain point

            totalslice = [item for sublist in totalslice for item in sublist]
            totalsslice = [item for sublist in totalsslice for item in
                           sublist]  # this can be treated as weights in samples

            if self.verbose > 0:
                end_time = time.time()
                print(f'Done in {(end_time - start_time)}.')
                start_time = time.time()
                print(f'Clustering data')
            X_bis_o = to_time_series_dataset(totalslice)
            X_bis_o_shap = to_time_series_dataset(totalsslice)

            print(X_bis_o.shape)
            print(X_bis_o_shap.shape)

            if not refit:
                if self.method != 'dtw':
                    cur_seq_len = X_bis_o.shape[1]
                    if cur_seq_len < self.seq_len:
                        diff = self.seq_len - cur_seq_len
                        if len(X_bis_o.shape) == 3:
                            expand = np.zeros((X_bis_o.shape[0], diff, X_bis_o.shape[2]))
                        elif len(X_bis.shape) == 2:
                            expand = np.zeros((X_bis_o.shape[0], diff))
                        X_bis = np.append(X_bis_o, expand, axis=1)
                        X_bis_shap = np.append(X_bis_o_shap, expand, axis=1)
                    elif cur_seq_len > self.seq_len:
                        diff = cur_seq_len - self.seq_len
                        X_bis = np.delete(X_bis_o, slice(X_bis_o.shape[1] - diff, None), 1)
                        X_bis_shap = np.delete(X_bis_o_shap, slice(X_bis_o_shap.shape[1] - diff, None), 1)
                    else:
                        X_bis = X_bis_o
                        X_bis_shap = X_bis_o_shap
                else:
                    X_bis = X_bis_o
                    X_bis_shap = X_bis_o_shap
            else:
                X_bis = X_bis_o
                X_bis_shap = X_bis_o_shap

            if self.method in ['kshape', 'tskshape', 'kshapegpu', 'kmeans']:
                if refit:
                    self.tsms_[dim] = TimeSeriesScalerMeanVariance()
                    X_bis = self.tsms_[dim].fit_transform(X_bis)
                else:
                    X_bis = self.tsms_[dim].transform(np.nan_to_num(X_bis))
            if self.method == 'kmeans':
                X_bis = X_bis.reshape(X_bis.shape[0], X_bis.shape[1])
                X_bis_shap = X_bis_shap.reshape(X_bis_shap.shape[0], X_bis_shap.shape[1])

            if self.verbose > 0:
                print(f'Shape of data for clustering: {X_bis.shape}')

            if refit:
                if self.verbose:
                    start_time = time.time()
                    print(f'Clustering data')
                if self.method == 'dtw':
                    self.kms_[dim] = TimeSeriesKMeans(n_clusters=self.n_clusters[self.feature_names[dim]], metric="dtw",
                                                      max_iter=5,
                                                      random_state=0, n_jobs=self.n_jobs)
                elif self.method == 'tskshape':
                    self.kms_[dim] = KShape(n_clusters=self.n_clusters[self.feature_names[dim]], verbose=True)
                elif self.method == 'kshape':
                    self.kms_[dim] = KShapeClusteringCPU(n_clusters=self.n_clusters[self.feature_names[dim]],
                                                         n_jobs=self.n_jobs)
                elif self.method == 'gpukshape':

                    self.kms_[dim] = KShapeClusteringGPU(n_clusters=self.n_clusters[self.feature_names[dim]])
                elif self.method == 'kmeans':
                    self.kms_[dim] = KMeans(n_clusters=self.n_clusters[self.feature_names[dim]])

                print(f'X_bis shape: {X_bis.shape} for method={self.method}')
                self.seq_len = X_bis.shape[1]

                self.kms_[dim].fit(X_bis)

                self.xbis_[self.feature_names[dim]] = X_bis
                self.xbis_shap_[self.feature_names[dim]] = X_bis_shap
                self.xbis_cluster_labels_[self.feature_names[dim]] = self.kms_[dim].labels_

            if self.verbose > 0:
                end_time = time.time()
                print(f'Done in {(end_time - start_time)}.')
                start_time = time.time()
                print(f'OHE time series')
            signal = np.concatenate(indexes)
            Xdf = pd.DataFrame(signal, columns=['sigid'])

            if refit:
                labels = self.kms_[dim].labels_
                self.label_features_[dim] = np.arange(0, np.max(labels) + 1)
                self.signal_ids_[self.feature_names[dim]] = np.concatenate(indexes)
            else:
                labels = self.kms_[dim].predict(X_bis)
                # todo xbis and cluster centers should be changed here?

            if not transform:
                return self

            Xdf['cluster'] = labels
            Xdf['shapweight'] = np.array([mean(abs(sv)) for sv in totalsslice])
            Xdf['durations'] = X_bis_o.shape[1] - np.sum(np.isnan(X_bis), axis=1)

            Xdf['min'] = np.nanmin(X_bis, axis=1)
            Xdf['max'] = np.nanmax(X_bis, axis=1)
            Xdf['mean'] = np.nanmean(X_bis, axis=1)
            Xdf['std'] = np.nanstd(X_bis, axis=1)
            Xdf['frequency'] = dominant_frequencies_for_rows(X_bis, sampling_rate=self.sampling_rate_)

            phantom = pd.DataFrame({'sigid': [-1] * len(self.label_features_[dim]),
                                    'cluster': np.arange(0, len(self.label_features_[dim])),
                                    'durations': [0] * len(self.label_features_[dim]),
                                    'min': [0] * len(self.label_features_[dim]),
                                    'max': [0] * len(self.label_features_[dim]),
                                    'mean': [0] * len(self.label_features_[dim]),
                                    'std': [0] * len(self.label_features_[dim]),
                                    'frequency': [0] * len(self.label_features_[dim])
                                    })

            Xdfp = pd.concat((Xdf, phantom))
            # print(f'Labels: {labels} label_features: {self.label_features_[dim]} and labels from km {np.unique(self.kms_[dim].labels_)}')

            ohe_train = pd.pivot_table(Xdfp, index='sigid', columns='cluster', values='durations',
                                       aggfunc=lambda x: sum(~np.isnan(x))).fillna(0).astype(int)
            duration_train = pd.pivot_table(Xdfp, index='sigid', values='durations', columns='cluster').fillna(
                0)  # TODO: wrong aggfunc
            min_train = pd.pivot_table(Xdfp, index='sigid', values='min', columns='cluster').fillna(0)
            max_train = pd.pivot_table(Xdfp, index='sigid', values='max', columns='cluster').fillna(0)
            mean_train = pd.pivot_table(Xdfp, index='sigid', values='mean', columns='cluster').fillna(0)
            std_train = pd.pivot_table(Xdfp, index='sigid', values='std', columns='cluster').fillna(0)
            frequency_train = pd.pivot_table(Xdfp, index='sigid', values='frequency', columns='cluster').fillna(0)
            self.weights_[dim] = Xdf.groupby('sigid')['shapweight'].mean().values

            # print(f'Columns: {ohe_train.columns}')

            ohe_train.columns = [f'exists_cl_{c}_{self.feature_names[dim]}' for c in
                                 range(0, len(self.label_features_[dim]))]
            duration_train.columns = [f'duration_cl_{c}_{self.feature_names[dim]}' for c in
                                      range(0, len(self.label_features_[dim]))]
            min_train.columns = [f'min_cl_{c}_{self.feature_names[dim]}' for c in
                                 range(0, len(self.label_features_[dim]))]
            max_train.columns = [f'max_cl_{c}_{self.feature_names[dim]}' for c in
                                 range(0, len(self.label_features_[dim]))]
            mean_train.columns = [f'mean_cl_{c}_{self.feature_names[dim]}' for c in
                                  range(0, len(self.label_features_[dim]))]
            std_train.columns = [f'std_cl_{c}_{self.feature_names[dim]}' for c in
                                 range(0, len(self.label_features_[dim]))]
            frequency_train.columns = [f'frequency_cl_{c}_{self.feature_names[dim]}' for c in
                                       range(0, len(self.label_features_[dim]))]

            train = pd.concat((ohe_train, duration_train, min_train, max_train, mean_train, std_train, frequency_train),
                              axis=1)

            train = train[~train.index.isin([-1])]

            if 'existance' in self.descriptors:
                features = features + list(ohe_train.columns)
            if 'duration' in self.descriptors:
                features = features + list(duration_train.columns)
            if 'stats' in self.descriptors:
                features = features + list(min_train.columns) + list(max_train.columns) + list(
                    mean_train.columns) + list(std_train.columns)
            if 'frequency' in self.descriptors:
                features = features + list(frequency_train.columns)

            train_dim.append(train)

        train = pd.concat(train_dim, axis=1)
        target = 'target'

        print(f'Len X={len(X)} vs len of train = {len(train)}')
        bbox_predictions = self.blackbox.predict(Xdim)
        if len(bbox_predictions.shape) == 2:
            train[target] = np.argmax(bbox_predictions, axis=1)
        else:
            train[target] = bbox_predictions
        if self.verbose > 0:
            end_time = time.time()
            print(f'Done in {(end_time - start_time)}.')
        weights = pd.DataFrame(self.weights_).sum(axis=1).values
        return train, features, target, weights


import numpy as np


def dominant_frequency(time_series_row, sampling_rate):
    """
    Calculate the dominant frequency of a time series row using FFT.

    Parameters:
    - time_series_row: 1D numpy array representing the time series data.
    - sampling_rate: The sampling rate of the time series.

    Returns:
    - dominant_freq: Dominant frequency of the time series row.
    """
    # Check for NaN values and replace them with zeros
    time_series_row = np.nan_to_num(time_series_row)

    # Compute the FFT
    fft_result = np.fft.fft(time_series_row)

    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(len(time_series_row), d=1 / sampling_rate)

    # Find the index of the maximum amplitude in the positive frequencies
    positive_freq_mask = frequencies > 0
    dominant_freq_index = np.argmax(np.abs(fft_result[positive_freq_mask]))

    # Get the dominant frequency
    dominant_freq = frequencies[positive_freq_mask][dominant_freq_index]

    return np.abs(dominant_freq)


def dominant_frequencies_for_rows(time_series_array, sampling_rate):
    """
    Calculate the dominant frequency for each row of a 2D numpy array.

    Parameters:
    - time_series_array: 2D numpy array where each row represents a time series.
    - sampling_rate: The sampling rate of the time series.

    Returns:
    - dominant_frequencies: 1D numpy array containing the dominant frequency for each row.
    """
    # Apply the dominant_frequency function to each row
    dominant_frequencies = np.apply_along_axis(dominant_frequency, axis=1, arr=time_series_array,
                                               sampling_rate=sampling_rate)

    return dominant_frequencies


def interpretable_model(ohe_train, features, target, intclf=None, verbose=0, max_depth=None, min_samples_leaf=0.05,
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


def plot_and_save_barplot(target, filename, figsize=(4, 2)):
    """
    Plots a bar plot of the number of instances per class and saves it to a file.

    Parameters:
    - target: list or array-like, the target array containing integer class labels.
    - filename: str, the filename to save the plot (including the .png extension).
    - figsize: tuple, optional, the size of the figure (width, height).

    Returns:
    None
    """
    # Use Set2 colormap directly by indexing with target values
    colors = plt.cm.Set2(np.unique(target))

    # Count the occurrences of each class
    class_counts = {label: list(target).count(label) for label in set(target)}
    # Extract class labels and counts

    if len(target) == 0:
        class_counts = {0: 0, 1: 0}
    classes, counts = zip(*class_counts.items())

    # Plotting the bar plot with Set2 colormap
    fig, ax = plt.subplots(figsize=figsize)
    bars = plt.bar(classes, counts, color=colors)
    plt.title('Number of Instances per Class')
    plt.xlabel('Class')
    plt.ylabel('Count')

    legend_labels = [f'Class {label}' for label in classes]
    plt.legend(bars, legend_labels, title='Legend')

    # Save the plot to the specified filename in PNG format
    fig.savefig(filename, format='png', bbox_inches='tight')
    plt.close(fig)


def plot_barycenter_with_histogram(X_train, X_train_sigids, X_train_shap, cluster_labels, cluster_centers,
                                   target_column=None, threshold=None,
                                   cluster_index=0, stat_function='max', human_friendly_name=None, ax=None,
                                   sampling_rate=1, figsize=(6, 3)):
    """
    Plots the barycenter of time series along with histograms of specified statistics in the specified cluster.
    If target_column is None, the function plots the histogram for the entire cluster statistics.

    Parameters:
    - X_train: Time series data.
    - y_pred: Predicted cluster labels.
    - cluster_centers: Cluster centers obtained from a clustering algorithm.
    - target_column: Target column (array-like) for class information. Default is None.
    - cluster_index: Index of the cluster to plot (default is 0).
    - stat_function: Statistic function for histogram ('max', 'min', 'std', 'mean', 'duration'). Default is 'max'.
    """
    # Extract the barycenter
    barycenter = cluster_centers[cluster_index]
    X_train_full = X_train.copy()
    # TODO: in fact if tarrget_column[cl=ci] contains only one class, we should change the stat_function to exists
    # if len(np.unique(target_column[cluster_labels == cluster_index])) < 2:
    #    stat_function='exists'

    # Extract values in the specified cluster based on the chosen statistic
    if stat_function == 'max':
        # todo: X_train is in fact Xbis with only one feature, need to median over sigid
        # xbisindices are needed to calculate correct stats
        cluster_values = np.nanmax(X_train[cluster_labels == cluster_index], axis=1)
        stat_name = "Maximum"
    elif stat_function == 'min':
        cluster_values = np.nanmin(X_train[cluster_labels == cluster_index], axis=1)
        stat_name = "Minimum"
    elif stat_function == 'std':
        cluster_values = np.nanstd(X_train[cluster_labels == cluster_index], axis=1)
        stat_name = "Standard Deviation"
    elif stat_function == 'mean':
        cluster_values = np.nanmean(X_train[cluster_labels == cluster_index], axis=1)
        stat_name = "Mean"
    elif stat_function == 'duration':
        cluster_values = X_train[cluster_labels == cluster_index].shape[1] - np.sum(
            np.isnan(X_train[cluster_labels == cluster_index]), axis=1)
        stat_name = "Duration"
    elif stat_function == 'frequency':
        cdata = X_train[cluster_labels == cluster_index]
        cluster_values = dominant_frequencies_for_rows(cdata, sampling_rate=sampling_rate)
        stat_name = 'Frequency'
    elif stat_function == 'exists':
        cluster_values = None
        stat_name = "Number of occurence"
    else:
        raise ValueError("Invalid stat_function. Choose from 'max', 'min', 'std', 'mean', or 'duration'.")

    Xdf = pd.DataFrame(cluster_values, columns=['cluster_values'])
    Xdf['sigid'] = X_train_sigids[cluster_labels == cluster_index]

    # nXdf = pd.DataFrame( X_train_sigids[~np.isin(X_train_sigids,Xdf['sigid'].values)], columns=['sigid'])

    Xdf['target'] = target_column[cluster_labels == cluster_index]
    if stat_function != 'exists':
        Xdfgr = Xdf.groupby('sigid')
        cluster_values = Xdfgr['cluster_values'].mean().values  # average maximum value. this could be median
        target_column = Xdfgr['target'].first().values
        X_train = X_train[cluster_labels == cluster_index]
    else:
        X_train = X_train[cluster_labels == cluster_index]
        target_column = target_column[cluster_labels == cluster_index]

    # Create the main figure and axis
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=figsize)

    ##No matter what, we want to plot exaples of time series with prototype in consideration
    sample_mask = X_train_sigids == Xdf['sigid'].iloc[0]
    full_sample_a = X_train_full[sample_mask]
    cluster_part = np.where(cluster_labels[sample_mask] == cluster_index)[0]
    full_sample = np.concatenate(full_sample_a)
    full_sample = full_sample[~np.isnan(full_sample)]
    ax[0].plot(full_sample, "black", alpha=0.5)
    print(cluster_labels[sample_mask])

    start_point = 0
    prev_color = None
    # Plot each array one after another
    for i, array in enumerate(full_sample_a):
        # Calculate x-axis values for the current array
        array = array[~np.isnan(array)]
        x_values = np.arange(start_point, start_point + len(array))

        # Plot the array with the specified color
        if i in cluster_part:
            ax[0].plot(x_values, array, label=f'Array {i + 1}', color='red')
            prev_color = 'red'
        else:
            prev_color = None

        # Update the starting point for the next array
        start_point = start_point + len(array)

        # print([x_values[-1], start_point])
        # print([array[-1], full_sample_a[i + 1].ravel()[0]])
        if i < len(full_sample_a) - 1 and prev_color is not None:
            ax[0].plot([x_values[-1], start_point], [array[-1], full_sample_a[i + 1].ravel()[0]], color=prev_color)

    print(
        f'Shapes: {X_train_full.shape} vs shap: {X_train_shap[sample_mask].shape} TODO: take IDS of a single sample that is plotted')
    shap_sample = np.concatenate(X_train_shap[sample_mask])
    reference_value_sampe = np.concatenate(full_sample_a)
    shap_sample = shap_sample[~np.isnan(reference_value_sampe)]
    ax[0].imshow([shap_sample], cmap='viridis', aspect='auto',
                 extent=[0, len(shap_sample), min(reference_value_sampe.ravel()), max(reference_value_sampe.ravel())],
                 alpha=0.3)

    # Contrastive examples (the one that does not have that cluster)
    # if there are some sigids that do not contain the cluster

    # get opposite class exmaples

    # if len(nXdf['sigid']) >0:
    #     sample_mask = X_train_sigids==nXdf['sigid'].iloc[0]
    #     full_sample_a = X_train_full[sample_mask]
    #     full_sample = np.concatenate(full_sample_a)
    #     full_sample=full_sample[~np.isnan(full_sample)]
    #     ax[0].plot(full_sample,"green", alpha=0.5)

    #     print(cluster_labels[sample_mask])
    # else:

    if target_column is not None:
        # Plot individual time series and barycenter
        for series, class_label in zip(X_train, target_column):
            ax[1].plot(series.ravel(), "k-", alpha=0.7, color=plt.cm.Set2(class_label))

        selected_indices = np.random.choice(X_train_full[cluster_labels != cluster_index].shape[0],
                                            min(100, X_train_full[cluster_labels != cluster_index].shape[0]),
                                            replace=False)
        for si in selected_indices:
            ax[1].plot(X_train_full[cluster_labels != cluster_index][si, :].ravel(), "k-", alpha=0.05)

        ax[1].plot(barycenter.ravel(), "r-", linewidth=2)
        ax[1].set_title(f"{stat_name} of cluster {cluster_index} of feature {human_friendly_name}")
        ax[1].legend()

        if stat_function != 'exists':
            # Create an inset axis for the histogram
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("bottom", size="20%", pad=0.1)

            # Plot histograms of specified statistic in the specified cluster for each class in the target column

            for class_label in np.unique(target_column):
                class_indices = np.where((target_column == class_label))[0]

                if len(class_indices) > 0:
                    class_values = cluster_values[class_indices]
                    cax.hist(class_values, bins=100, alpha=0.7, label=f"Class {class_label}",
                             color=plt.cm.Set2(class_label))

            if threshold is not None:
                cax.axvline(threshold, linestyle='--', color='r')
            # cax.set_title(f"Histograms of {stat_name} Values (Cluster {cluster_index} of feature {human_friendly_name} )")
            cax.set_xlabel(stat_name)
            cax.set_ylabel("Frequency")
            cax.legend(loc='upper right')

    else:
        # If target_column is None, plot the individual time series on the barycenter plot without coloring
        for series in X_train:
            ax[1].plot(series.ravel(), "k-", alpha=0.7)
        ax[1].plot(barycenter.ravel(), "r-", linewidth=2,
                   label=f"Barycenter (Cluster {cluster_index} of feature {human_friendly_name})")
        ax[1].set_title(f"Barycenter of Time Series (Cluster {cluster_index} of feature {human_friendly_name})")
        ax[1].legend()

        if stat_function != 'exists':
            # Create an inset axis for the histogram
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("bottom", size="20%", pad=0.1)

            # Plot histogram of specified statistic in the specified cluster
            cax.hist(cluster_values, bins=100, color='blue', alpha=0.7)
            if threshold is not None:
                cax.axvline(threshold, linestyle='--', color='r')
            # cax.set_title(f"Histogram of {stat_name} Values (Cluster {cluster_index} of feature {human_friendly_name})")
            cax.set_xlabel(stat_name)
            cax.set_ylabel("Frequency")


def plot_histogram(ax, dataset, xbis, xbis_shap, xbisclusters, xbisindices, feature_name, target_name, threshold=None,
                   proto_encoder=None):
    """
    Plot a histogram colored by the values of the target.

    Parameters:
    - ax: Matplotlib axis object
    - data: Data array to plot
    - feature_name: Name of the feature for labeling the plot
    - target_name: Name of the target variable for labeling the plot
    """

    (stat_function, _, cluster_index, human_friendly_name) = feature_name.split('_', 3)

    all_features = [f for f in dataset.columns if f not in [target_name]]
    cluster_centers = proto_encoder.kms_[proto_encoder.feature_names.index(human_friendly_name)].cluster_centers_

    signaldf = pd.DataFrame(xbisindices[human_friendly_name], columns=['sigid']).set_index('sigid')
    target_values = signaldf.join(dataset[[target_name]], how='right').dropna().values

    # print(f'cbis shape:{xbis[human_friendly_name].shape}')
    # print(f'target shap  = {target_values.shape}')
    # print(f'xbisclusters shape = {xbisclusters[human_friendly_name].shape}')
    plot_barycenter_with_histogram(X_train=xbis[human_friendly_name], X_train_sigids=xbisindices[human_friendly_name],
                                   cluster_labels=xbisclusters[human_friendly_name],
                                   X_train_shap=xbis_shap[human_friendly_name],
                                   cluster_centers=cluster_centers, target_column=target_values,
                                   # target has to be populated over sigid
                                   cluster_index=int(cluster_index), stat_function=stat_function, ax=ax,
                                   human_friendly_name=human_friendly_name,
                                   sampling_rate=proto_encoder.sampling_rate_, threshold=threshold)

    # ax.legend()


def save_histogram_svg(dataset, xbis, xbis_shap, xbisclusters, xbisindices, feature_name, target_name, threshold,
                       filename, proto_encoder=None, figsize=(6, 3)):
    """
    Save the histogram plot as an SVG file.

    Parameters:
    - data: Data array to plot
    - feature_name: Name of the feature for labeling the plot
    - filename: Name of the SVG file to save
    """
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    plot_histogram(ax, dataset, xbis, xbis_shap, xbisclusters, xbisindices, feature_name, target_name,
                   threshold=threshold, proto_encoder=proto_encoder)
    fig.savefig(filename, format='png', bbox_inches='tight')
    plt.close(fig)


def get_node_histogram_svg_filenames(decision_tree, node, dataset, xbis, xbis_shap, xbisclusters, xbisindices,
                                     target_name, feature_names, output_dir, proto_encoder=None, stat_nodes={},
                                     figsize=(6, 3)):
    """
    Save SVG representations of histograms for a given internal node.

    Parameters:
    - decision_tree: DecisionTreeClassifier object
    - node: Node index
    - feature_names: List of feature names
    - output_dir: Directory to save SVG files

    Returns:
    - Tuple (left_histogram_filename, right_histogram_filename)
    """
    # feature_name = feature_names[decision_tree.tree_.feature[node]]
    # left_data = decision_tree.tree_.value[decision_tree.tree_.children_left[node]].ravel()
    # right_data = decision_tree.tree_.value[decision_tree.tree_.children_right[node]].ravel()

    # left_svg_filename = os.path.join(output_dir, f'left_{node}.svg')
    # right_svg_filename = os.path.join(output_dir, f'right_{node}.svg')

    # save_histogram_svg(left_data, f'{feature_name} <= {decision_tree.tree_.threshold[node]:.2f}', left_svg_filename)
    # save_histogram_svg(right_data, f'{feature_name} > {decision_tree.tree_.threshold[node]:.2f}', right_svg_filename)
    feature_name = feature_names[decision_tree.tree_.feature[node]]

    # TODO: this should simply contain summary of classes
    svg_filename = os.path.join(output_dir, f'{node}.png')

    if decision_tree.tree_.children_left[node] != decision_tree.tree_.children_right[node]:
        save_histogram_svg(dataset, xbis, xbis_shap, xbisclusters, xbisindices, feature_name, target_name,
                           decision_tree.tree_.threshold[node], svg_filename, proto_encoder=proto_encoder,
                           figsize=figsize)
    else:
        plot_and_save_barplot(dataset[target_name].values, svg_filename)

    embedded_html = f'<<table border="0" cellborder="0"><tr><td><img src="{svg_filename}" /></td></tr></table>>'

    left_cond = f'`{feature_name}` <= {decision_tree.tree_.threshold[node]}'
    right_cond = f'`{feature_name}` > {decision_tree.tree_.threshold[node]}'

    combined = [(node, embedded_html)]

    def filter_xbis(xbis, xbisindices, ds):
        new_xbis = {}
        for k in xbis.keys():
            indices = np.where(np.isin(xbisindices[k], list(ds.index)))[0]
            new_xbis[k] = xbis[k][indices]
        return new_xbis

    # print(f' feature name {feature_name}, left: {decision_tree.tree_.children_left[node]} right {decision_tree.tree_.children_right[node]}')

    if decision_tree.tree_.children_left[node] != -1:
        leftds = dataset.query(left_cond)
        new_xbis = filter_xbis(xbis, xbisindices, leftds)
        new_xbisclusters = filter_xbis(xbisclusters, xbisindices, leftds)
        new_xbisindices = filter_xbis(xbisindices, xbisindices, leftds)
        nex_xbisshap = filter_xbis(xbis_shap, xbisindices, leftds)
        left_dot = get_node_histogram_svg_filenames(decision_tree, decision_tree.tree_.children_left[node],
                                                    dataset.query(left_cond), new_xbis, nex_xbisshap, new_xbisclusters,
                                                    new_xbisindices,
                                                    target_name, feature_names, output_dir,
                                                    proto_encoder=proto_encoder)  #
        combined += left_dot
    if decision_tree.tree_.children_right[node] != -1:
        rightds = dataset.query(right_cond)
        new_xbis = filter_xbis(xbis, xbisindices, rightds)
        new_xbisclusters = filter_xbis(xbisclusters, xbisindices, rightds)
        new_xbisindices = filter_xbis(xbisindices, xbisindices, rightds)
        nex_xbisshap = filter_xbis(xbis_shap, xbisindices, rightds)
        right_dot = get_node_histogram_svg_filenames(decision_tree, decision_tree.tree_.children_right[node],
                                                     dataset.query(right_cond), new_xbis, nex_xbisshap,
                                                     new_xbisclusters, new_xbisindices,
                                                     target_name, feature_names, output_dir,
                                                     proto_encoder=proto_encoder)  #
        combined += right_dot

    return combined


def embed_histograms_in_dot(decision_tree, dataset, xbis, xbis_shap, xbisclusters, xbisindices, target_name,
                            feature_names, output_dir, proto_encoder=None, figsize=(6, 3)):
    """
    Embed histograms in DOT format for each internal node.

    Parameters:
    - decision_tree: DecisionTreeClassifier object
    - feature_names: List of feature names
    - output_dir: Directory to save SVG files

    Returns:
    - Digraph object with embedded histograms
    """
    dot = Digraph(comment='Decision Tree with Histograms')
    dot.format = 'png'

    combined_dot = get_node_histogram_svg_filenames(decision_tree, 0, dataset, xbis, xbis_shap, xbisclusters,
                                                    xbisindices, target_name, feature_names, output_dir,
                                                    proto_encoder=proto_encoder, figsize=figsize)

    for (node, embedded_html) in combined_dot:
        dot.node(str(node), label=embedded_html, _attributes={'shape': 'record'})

    for i in range(decision_tree.tree_.node_count):
        left_child = decision_tree.tree_.children_left[i]
        right_child = decision_tree.tree_.children_right[i]

        ##think on some change in case of "existance"
        # TODO: change threshold if exists is a fuction
        if left_child != right_child:  # Internal node
            dot.edge(str(i), str(left_child), label=f'<= {decision_tree.tree_.threshold[i]:.2f}')
            dot.edge(str(i), str(right_child), label=f'> {decision_tree.tree_.threshold[i]:.2f}')

    return dot


def export_decision_tree_with_embedded_histograms(decision_tree, dataset, target_name, feature_names, filename,
                                                  proto_encoder=None, figsize=(6, 3)):
    """
    Export a Decision Tree classifier to a DOT file with embedded histograms at each node.

    Parameters:
    - decision_tree: DecisionTreeClassifier object
    - feature_names: List of feature names
    - filename: Name of the DOT file to save
    """
    output_dir = 'histograms'
    os.makedirs(output_dir, exist_ok=True)

    if True:  # try:
        xbis = proto_encoder.xbis_
        xbisclusters = proto_encoder.xbis_cluster_labels_
        xbisindices = proto_encoder.signal_ids_
        xbis_shap = proto_encoder.xbis_shap_
        dot = embed_histograms_in_dot(decision_tree, dataset, xbis, xbis_shap, xbisclusters, xbisindices, target_name,
                                      feature_names, output_dir, proto_encoder=proto_encoder, figsize=figsize)
        # dot.render(filename, cleanup=True, view=True)
        dot.render(filename, cleanup=True)
        print(f"Decision Tree exported to {filename} with embedded histograms successfully.")
        return dot

    # except Exception as e:
    #    print(f"Error exporting Decision Tree to DOT file with embedded histograms: {e}")
    return None

# Example usage:
# Assuming you have a DecisionTreeClassifier object named 'dt_classifier'
# feature_names is the list of feature names, and you want to save the DOT file as 'decision_tree'
# export_decision_tree_with_embedded_histograms(dt_classifier, feature_names, 'decision_tree')


