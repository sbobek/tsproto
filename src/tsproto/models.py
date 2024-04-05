import warnings
import ruptures as rpt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.tree import DecisionTreeClassifier
import time
from tslearn.clustering import KShape
from kshape.core import KShapeClusteringCPU
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from tsproto.utils import dominant_frequencies_for_rows

"""
Documentation of this module

"""


class PrototypeEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes time-series into prototypes
    """

    def __init__(self, blackbox, min_size, jump, pen, n_clusters, multiplier=1.5, method='dtw',
                 descriptors=['existance', 'duration', 'stats', 'startpoint'], n_jobs=None, verbose=0,
                 dims=1, sampling_rate=1, feature_names=None, importance_aggregation_func=np.mean):
        """ Initializes PrototypeEncoder class

        :param blackbox: instance of a blackbox model that is to be explained
        :param min_size: minimum size of a prototype (Pelt algorithm parameter)
        :param jump: subsample (one every jump points) (Pelt algorithm parameter)
        :param pen: penalty value (>0) (Pelt algorithm parameter)
        :param n_clusters: number of clusters to generated (these are going to be prototypes). It can be int, float or dict.
        :param multiplier: multiplier used in outlier deteciton. The smaller the value the stronger reduction of outliers
        :param method: clustering algorithms method. Default dtw.
        :param descriptors: what description functions use to describe prototypes.
        :param n_jobs: parallelization. Default None
        :param verbose: verbosity level. Default 0
        :param dims: number of dimensions/features in time series. Default 1
        :param sampling_rate: sampling rate used to calculate dominant frequency. Default 1
        :param feature_names: list of feature names. Default None
        :param importance_aggregation_func: function to aggregate shap values. Default np.mean
        """
        self.threshold = 0  # threshold for discarding slices of time-series of low importance
        self.multiplier = multiplier
        self.n_clusters = {}
        self.jump = jump
        self.min_size = min_size
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pen = pen
        self.dims = dims
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
        self.sampling_rate = sampling_rate
        self.weights_ = {}
        self.seq_len_ = {}
        self.importance_aggregation_func = importance_aggregation_func

        if self.method == 'kshapegpu':
            try:
                from kshape.core_gpu import KShapeClusteringGPU
            except ImportError:
                self.method = 'kshape'
                warnings.warn(
                    "To use GPU version of KSHAPE, install TSProto with GPU support: pip install tsproto[gpu]")

        if feature_names is None:
            self.feature_names = [str(f) for f in range(0, dims)]
        else:
            self.feature_names = feature_names

        # Make the number of clusters be determined by the number of breakpoints in data
        if n_clusters is None:
            n_clusters = 1

        # If the n_clusters was given but it is not dict, assign same number of clusters to all
        # dmiensions. It can be a fraction, which will make the algorithm determine the optimal number of clusters
        # while fitting. The fraction represetns the desired granularity of clustersng.
        # Otherwise, just assigne the dict to self.n_clusters this allows for a mixture of
        # fixed number of clusters and fractions (automatic detection)
        if not isinstance(n_clusters, dict):
            for dim in range(dims):
                self.n_clusters[self.feature_names[dim]] = n_clusters
        else:
            self.n_clusters = n_clusters

    def fit(self, X, y=None, shapclass=None):
        """ Fits model to the training data

        :param X:
        :type X: array-like, shape (n_samples, n_timestamps, n_features)
        :param y: not used, kept only for the compliance with scikit-learn API
        :type y: array-like
        :param shapclass: SHAP value calculated for the blackbox model to be explained.
        It is assumed that the mean absolute values are passed here.
        :type shapclass: array-like, shape (n_samples,n_timestamps,)
        :return:
        """
        if y is not None and shapclass is None:
            shapclass = y
        return self._transform(X, shapclass, refit=True, transform=False)

    def transform(self, X, y=None, shapclass=None):
        """

        :param X:
        :param y:
        :param shapclass:
        :return:
        """

        if y is not None and shapclass is None:
            shapclass = y
        return self._transform(X, shapclass, refit=False, transform=True)

    def fit_transform(self, X, y=None, shapclass=None):
        """

        :param X:
        :param y:
        :param shapclass:
        :return:
        """
        if y is not None and shapclass is None:
            shapclass = y
        return self._transform(X, shapclass, refit=True, transform=True)

    def _transform(self, Xdim, shap_dim, refit=False, transform=True):
        """

        :param Xdim:
        :param shap_dim:
        :param refit:
        :param transform:
        :return:
        """
        train_dim = []
        features = []
        if self.verbose > 0:
            print(f'Dataset shape: {Xdim.shape}')
            start_time = time.time()
            print(f'Calculating changepoints...')

        for dim in range(0, Xdim.shape[2]):
            totalslice = []
            totalsslice = []
            totalbpoints = []
            indexes = []
            print(f'Dim: {dim}')
            X = Xdim[:, :, dim].reshape((Xdim.shape[0], Xdim.shape[1], 1))
            shapclass = shap_dim[:, :, dim].reshape((shap_dim.shape[0], shap_dim.shape[1], 1))
            sum_bkps = 0
            for i in range(0, X.shape[0]):
                algo = rpt.Pelt(model="rbf", min_size=self.min_size, jump=self.jump).fit(shapclass[i])

                try:
                    breakpoints = algo.predict(pen=1)
                except:
                    # no change detected, means there are no breakpoints
                    breakpoints = [0]

                sum_bkps += len(breakpoints)
                shapslices = [abs(s) for s in np.split(shapclass[i], breakpoints) if
                              len(s) != 0]  # (assuming this is abshap over all classses)
                slices = [s for s in np.split(X[i], breakpoints) if len(s) != 0]
                bpoints = list(set([0] + [s for s in breakpoints if s < len(X[i]) and s > 0]))

                totalsslice.append(shapslices)
                totalslice.append(slices)
                totalbpoints.append(bpoints)
                indexes.append(np.ones(len(slices)) * i)

            print(len(totalsslice[0]))
            print(len(totalslice[0]))
            print(len(indexes[0]))
            print(len(totalbpoints[0]))

            ############ overriding clusters\
            if refit and (self.n_clusters[self.feature_names[dim]] <= 1):
                self.n_clusters[self.feature_names[dim]] = max(2, int(int(sum_bkps / X.shape[0]) * self.n_clusters[
                    self.feature_names[dim]]))
                print(f'For {self.feature_names[dim]} c_clusters = {self.n_clusters[self.feature_names[dim]]}')

            ############# .
            if refit:
                if self.multiplier is not None:
                    mean_concat_sslices = np.array(
                        [self.importance_aggregation_func(item) for sublist in totalsslice for item in sublist])
                    self.thresholds_[dim] = np.mean(mean_concat_sslices) - self.multiplier * np.std(mean_concat_sslices)
                else:
                    self.thresholds_[dim] = 0

            # ante-hoc filtering
            print(f'Before filtering no sigids: {(len(np.unique(np.concatenate(indexes))))}, uniqu {0}')
            if self.thresholds_[dim] > 0:
                indexed_slices = [[i, s, ss, bp] for ts, tss, ti, bpi in zip(totalslice, totalsslice, indexes, totalbpoints)
                                  for s, ss, i, bp in zip(ts, tss, ti, bpi) if
                                  self.importance_aggregation_func(abs(ss)) >= self.thresholds_[dim]]
                isdf = pd.DataFrame(indexed_slices)
                indexes_new = isdf.loc[:, 0]
                print(
                    f'Unique indexes: {len(np.unique(indexes_new))} vs max+1: {(len(np.unique(np.concatenate(indexes))))}')
                if len(np.unique(indexes_new)) < (len(np.unique(np.concatenate(indexes)))):
                    # in case there is a signal that has absolutely no readings
                    # reduce the threshold
                    print('WARNING: Changing the threshold, due to empty record')
                    full_slices = [[i, s, ss, bp] for ts, tss, ti, bpi in zip(totalslice, totalsslice, indexes, totalbpoints)
                                   for s, ss, i, bp in zip(ts, tss, ti, bpi)]
                    isdf = pd.DataFrame(full_slices, columns=['index', 'slice', 'shapslice', 'breakpoint'])
                    isdf['maxshap'] = isdf['shapslice'].apply(lambda x: np.mean(abs(x)))
                    self.thresholds_[dim] = isdf.groupby('index')['maxshap'].max().min()
                    indexed_slices = [[i, s, ss, bp] for ts, tss, ti, bpi in
                                      zip(totalslice, totalsslice, indexes, totalbpoints) for s, ss, i, bp in
                                      zip(ts, tss, ti, bpi) if
                                      self.importance_aggregation_func(abs(ss)) >= self.thresholds_[dim]]
                    print(f'Threshold: {self.thresholds_[dim]}')
                    isdf = pd.DataFrame(indexed_slices)
                indexes = list(isdf.groupby(0)[0].apply(np.array))
                totalslice = list(isdf.groupby(0)[1].apply(list))
                totalsslice = list(isdf.groupby(0)[2].apply(list))
                totalbpoints = list(isdf.groupby(0)[3].apply(np.array))

            ############# post-hoc filtering
            # TODO: filter whole clusters which average/max importance is below certain point
            print(f'After removing , no sigids: {(len(np.unique(np.concatenate(indexes))))}, uniqu {0}')

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
                    if cur_seq_len < self.seq_len_[dim]:
                        diff = self.seq_len_[dim] - cur_seq_len
                        if len(X_bis_o.shape) == 3:
                            expand = np.zeros((X_bis_o.shape[0], diff, X_bis_o.shape[2]))
                        elif len(X_bis.shape) == 2:
                            expand = np.zeros((X_bis_o.shape[0], diff))
                        X_bis = np.append(X_bis_o, expand, axis=1)
                        X_bis_shap = np.append(X_bis_o_shap, expand, axis=1)
                    elif cur_seq_len > self.seq_len_[dim]:
                        diff = cur_seq_len - self.seq_len_[dim]
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

            if self.method in ['tskshape', 'kmeans']:
                X_bis = np.nan_to_num(X_bis)
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
                elif self.method == 'kshapegpu':
                    from kshape.core_gpu import KShapeClusteringGPU
                    self.kms_[dim] = KShapeClusteringGPU(n_clusters=self.n_clusters[self.feature_names[dim]])
                elif self.method == 'kmeans':
                    self.kms_[dim] = KMeans(n_clusters=self.n_clusters[self.feature_names[dim]])

                print(f'X_bis shape: {X_bis.shape} for method={self.method}')
                self.seq_len_[dim]=X_bis.shape[1]

                self.kms_[dim].fit(X_bis)

                if self.method == 'kshape':
                    self.kms_[dim].cluster_centers_ = self.kms_[dim].centroids_
                if self.method == 'kshapegpu':
                    self.kms_[dim].cluster_centers_ = self.kms_[dim].centroids_.detach().cpu()

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
            Xdf['shapweight'] = np.array([self.importance_aggregation_func(abs(sv)) for sv in totalsslice])
            Xdf['startpoint'] = np.concatenate(totalbpoints)
            Xdf['durations'] = X_bis_o.shape[1] - np.sum(np.isnan(X_bis), axis=1)

            Xdf['min'] = np.nanmin(X_bis, axis=1)
            Xdf['max'] = np.nanmax(X_bis, axis=1)
            Xdf['mean'] = np.nanmean(X_bis, axis=1)
            Xdf['std'] = np.nanstd(X_bis, axis=1)
            Xdf['frequency'] = dominant_frequencies_for_rows(X_bis, sampling_rate=self.sampling_rate)

            phantom = pd.DataFrame({'sigid': [-1] * len(self.label_features_[dim]),
                                    'cluster': np.arange(0, len(self.label_features_[dim])),
                                    'startpoint': np.arange(0, len(self.label_features_[dim])),
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
            startpoint_train = pd.pivot_table(Xdfp, index='sigid', columns='cluster', values='startpoint',
                                              aggfunc='min').fillna(0).astype(int)
            duration_train = pd.pivot_table(Xdfp, index='sigid', values='durations', columns='cluster').fillna(
                0)  # TODO: wrong aggfunc?
            min_train = pd.pivot_table(Xdfp, index='sigid', values='min', columns='cluster').fillna(0)
            max_train = pd.pivot_table(Xdfp, index='sigid', values='max', columns='cluster').fillna(0)
            mean_train = pd.pivot_table(Xdfp, index='sigid', values='mean', columns='cluster').fillna(0)
            std_train = pd.pivot_table(Xdfp, index='sigid', values='std', columns='cluster').fillna(0)
            frequency_train = pd.pivot_table(Xdfp, index='sigid', values='frequency', columns='cluster').fillna(0)
            self.weights_[dim] = Xdf.groupby('sigid')['shapweight'].apply(self.importance_aggregation_func).values
            print(f'Shape weights: {self.weights_[dim].shape}')

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
            startpoint_train.columns = [f'startpoint_cl_{c}_{self.feature_names[dim]}' for c in
                                        range(0, len(self.label_features_[dim]))]

            train = pd.concat((ohe_train, duration_train, min_train, max_train, mean_train, std_train, frequency_train,
                               startpoint_train), axis=1)
            train = train[~train.index.isin([-1])]

            if 'existance' in self.descriptors:
                features = features + list(ohe_train.columns)
            if 'duration' in self.descriptors:
                features = features + list(duration_train.columns) + list(startpoint_train.columns)
            if 'stats' in self.descriptors:
                features = features + list(min_train.columns) + list(max_train.columns) + list(
                    mean_train.columns) + list(std_train.columns)
            if 'frequency' in self.descriptors:
                features = features + list(frequency_train.columns)

            train_dim.append(train)

        train = pd.concat(train_dim, axis=1).fillna(0)
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
        weights = pd.DataFrame(pad_arrays_in_dict(self.weights_)).sum(axis=1).values
        return train, features, target, weights


def pad_arrays_in_dict(array_dict):
    # Find the maximum length among the arrays
    max_length = max(len(array) for array in array_dict.values())

    # Pad arrays with zeros to make them of equal length
    padded_arrays = {label: np.pad(array, (0, max_length - len(array)), mode='constant', constant_values=0)
                     for label, array in array_dict.items()}

    return padded_arrays


class InterpretableModel:
    def fit_or_predict(self, ohe_train, features, target, intclf=None, verbose=0, max_depth=None, min_samples_leaf=0.05,
                       weights=None, average=None):
        if intclf is None:
            intclf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            intclf.fit(ohe_train[features], ohe_train[target], sample_weight=weights)
        iclf_pred = intclf.predict(ohe_train[features])
        if verbose > 0:
            print(f'Tree depth: {intclf.get_depth()}')
        if average is None and max(len(np.unique(iclf_pred)), ohe_train[target].nunique()) > 2:
            average = 'macro'
        elif average is None:
            average = 'binary'
        return (accuracy_score(ohe_train[target], iclf_pred),
                precision_score(ohe_train[target], iclf_pred, average=average),
                recall_score(ohe_train[target], iclf_pred, average=average),
                f1_score(ohe_train[target], iclf_pred, average=average),
                intclf)
