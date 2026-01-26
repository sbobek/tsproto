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
from tsproto.utils import dominant_frequencies_for_rows, calculate_trends, cdist_kshape
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import kmedoids
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested

max_float32 = np.finfo(np.float32).max
min_float32 = np.finfo(np.float32).min

"""
Documentation of this module

"""


class PrototypeEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes time-series into prototypes
    """

    def __init__(self, blackbox, min_size, jump, pen, n_clusters, multiplier=1.5, global_breakpointing=False,
                 method='kshape', descriptors=['existance', 'duration', 'stats','similarity'],
                 n_jobs=None, verbose=0, dims=1, sampling_rate=1, feature_names=None,
                 importance_aggregation_func=np.mean, pelt_model='rbf'):
        """ Initializes PrototypeEncoder class

        :param blackbox: instance of a blackbox model that is to be explained
        :param min_size: minimum size of a prototype (Pelt algorithm parameter)
        :param jump: subsample (one every jump points) (Pelt algorithm parameter)
        :param pen: penalty value (>0)  for Pelt algorithm parameter.
        :param n_clusters: number of clusters to generated (these are going to be prototypes). It can be int, float or dict.
        If float, the number of clusters is determined dynamically as a product of the parameter and the average number of breakpoints detected in the particualr dimension.
        :param pelt_model: model for Pelt changepoint detection it can be l1, l2, or rbf.
        :param multiplier: multiplier used in outlier detection. The smaller the value the stronger reduction of outliers
        :param method: clustering algorithms method. Default dtw. Possible options: ['dtw','kshape','tskshape','rocket','kmeans','rocket']
        :param descriptors: what description functions use to describe prototypes.
        :param n_jobs: parallelization. Default None
        :param verbose: verbosity level. Possible values 0, 1, 2.  Default 0
        :param dims: number of dimensions/features in time series. Default 1
        :param sampling_rate: sampling rate used to calculate dominant frequency. Default 1
        :param feature_names: list of feature names. Default None
        :param importance_aggregation_func: function to aggregate shap values. Default np.mean
        """

        self.threshold = 0  # threshold for discarding slices of time-series of low importance
        self.multiplier = multiplier
        self.jump = jump
        self.min_size = min_size
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_clusters = {}
        self.pen = pen
        self.blackbox = blackbox
        self.descriptors = descriptors
        self.thresholds_ = [None] * dims
        self.kms_ = [None] * dims
        self.label_features_ = [None] * dims
        self.tsms_ = [None] * dims
        self.xbis_ = {}
        self.xbis_shap_ = {}
        self.xbis_ordidx_ = {}
        self.xbis_init_ = {}
        self.xbis_shap_init_ = {}
        self.xbis_cluster_labels_ = {}
        self.sbc_ = {}
        self.signal_ids_ = {}
        self.sampling_rate_ = sampling_rate
        self.weights_ = {}
        self.seq_len_ = {}
        self.pelt_model = pelt_model
        self.importance_aggregation_func = importance_aggregation_func
        self.pen = pen  # FX3
        self.cluster_override_factor = {}
        self.global_breakpointing = global_breakpointing

        if self.method == 'sbc' and not self.global_breakpointing:
            warnings.warn(
                "Overriding parameter global_breakpointing to True, as sbc method cannot work with local breakpoints.")
            self.global_breakpointing = True

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

        bbox_predictions = self.blackbox.predict(Xdim)
        if len(bbox_predictions.shape) == 2:
            bbox_label = np.argmax(bbox_predictions, axis=1)
        else:
            bbox_label = bbox_predictions
        for dim in range(0, Xdim.shape[2]):
            totalslice = []
            totalsslice = []
            totalbpoints = []
            indexes = []
            bbox_preds = []
            ordidx = []
            X = Xdim[:, :, dim].reshape((Xdim.shape[0], Xdim.shape[1], 1))
            shapclass = shap_dim[:, :, dim].reshape((shap_dim.shape[0], shap_dim.shape[1], 1))
            sum_bkps = 0

            if self.global_breakpointing:
                if refit:
                    avhap = np.mean(shapclass, axis=0)
                    algo = rpt.Pelt(model=self.pelt_model, min_size=self.min_size, jump=self.jump).fit(avhap)
                    self.globalbpoints = algo.predict(pen=self.pen)
                totalbpoints = [self.globalbpoints for _ in range(0, X.shape[0])]
            else:
                for i in range(0, X.shape[0]):
                    algo = rpt.Pelt(model=self.pelt_model, min_size=self.min_size, jump=self.jump).fit(shapclass[i])
                    try:
                        breakpoints = algo.predict(pen=self.pen)
                    except:
                        breakpoints = [0]
                    totalbpoints.append(breakpoints)

            for i in range(0, X.shape[0]):
                breakpoints = totalbpoints[i]
                sum_bkps += len(breakpoints)
                shapslices = [abs(s) for s in np.split(shapclass[i], breakpoints) if
                              len(s) != 0]  # (assuming this is abshap over all classses)
                slices = [s for s in np.split(X[i], breakpoints) if len(s) != 0]
                bpoints = [0] + [s for s in breakpoints if s < len(X[i]) and s > 0]

                totalsslice.append(shapslices)
                totalslice.append(slices)
                indexes.append(np.ones(len(slices)) * i)
                bbox_preds.append(np.ones(len(slices)) * bbox_label[i])
                ordidx.append(np.arange(0, len(bpoints)))

            self.xbis_init_[self.feature_names[dim]] = totalslice
            self.xbis_shap_init_[self.feature_names[dim]] = totalsslice

            ############ overriding clusters
            if refit and (self.n_clusters[self.feature_names[dim]] <= 1 or isinstance(
                    self.n_clusters[self.feature_names[dim]], float)):  # FX 5
                self.cluster_override_factor[self.feature_names[dim]] = self.n_clusters[self.feature_names[dim]]
                self.n_clusters[self.feature_names[dim]] = max(2, int(sum_bkps / X.shape[0] * self.n_clusters[
                    self.feature_names[dim]] + 0.5))
                if self.verbose > 0:
                    print(f'For {self.feature_names[dim]} c_clusters = {self.n_clusters[self.feature_names[dim]]}, based on average bkps: {sum_bkps / X.shape[0]}')
            elif refit:
                self.cluster_override_factor[self.feature_names[dim]] = None

            #############

            if refit:
                if self.multiplier is not None:
                    mean_concat_sslices = np.array(
                        [self.importance_aggregation_func(item) for sublist in totalsslice for item in sublist])
                    self.thresholds_[dim] = np.mean(mean_concat_sslices) - self.multiplier * np.std(
                        mean_concat_sslices)
                else:
                    self.thresholds_[dim] = 0

            # ante-hoc filtering
            if self.thresholds_[dim] > 0 and self.method != 'sbc':
                indexed_slices = [[i, s, ss, bp, bbp, o] for ts, tss, ti, bpi, bbpi, oi in
                                  zip(totalslice, totalsslice, indexes, totalbpoints, bbox_preds, ordidx) for
                                  s, ss, i, bp, bbp, o in zip(ts, tss, ti, bpi, bbpi, oi) if
                                  self.importance_aggregation_func(abs(ss)) >= self.thresholds_[dim]]
                isdf = pd.DataFrame(indexed_slices)
                indexes_new = isdf.loc[:, 0] if len(isdf) > 0 else np.array([])
                if len(np.unique(indexes_new)) < (len(np.unique(np.concatenate(indexes)))):
                    # in case there is a signal that has absolutely no readings
                    # reduce the threshold
                    print('WARNING: Changing the threshold, due to empty record')
                    full_slices = [[i, s, ss, bp, bbp, o] for ts, tss, ti, bpi, bbpi, oi in
                                   zip(totalslice, totalsslice, indexes, totalbpoints, bbox_preds, ordidx) for
                                   s, ss, i, bp, bbp, o in zip(ts, tss, ti, bpi, bbpi, oi)]
                    isdf = pd.DataFrame(full_slices,
                                        columns=['index', 'slice', 'shapslice', 'breakpoint', 'bbox_preds',
                                                 'ordidx'])
                    isdf['maxshap'] = isdf['shapslice'].apply(
                        lambda x: self.importance_aggregation_func(abs(x)))  # FX2
                    self.thresholds_[dim] = isdf.groupby('index')['maxshap'].max().min()
                    indexed_slices = [[i, s, ss, bp, bbp, o] for ts, tss, ti, bpi, bbpi, oi in
                                      zip(totalslice, totalsslice, indexes, totalbpoints, bbox_preds, ordidx) for
                                      s, ss, i, bp, bbp, o in zip(ts, tss, ti, bpi, bbpi, oi) if
                                      self.importance_aggregation_func(abs(ss)) >= self.thresholds_[dim]]
                    isdf = pd.DataFrame(indexed_slices)
                indexes = list(isdf.groupby(0)[0].apply(np.array))
                totalslice = list(isdf.groupby(0)[1].apply(list))
                totalsslice = list(isdf.groupby(0)[2].apply(list))
                totalbpoints = list(isdf.groupby(0)[3].apply(np.array))
                bbox_preds = list(isdf.groupby(0)[4].apply(np.array))
                ordidx = list(isdf.groupby(0)[5].apply(np.array))
            elif not refit and self.thresholds_[dim] > 0 and self.method == 'sbc':
                # simply silence down  slices that are not in ordix
                indexes = [[sublist[i] for i in self.kms_[dim].ordidx] for sublist in indexes]
                totalslice = [[sublist[i] for i in self.kms_[dim].ordidx] for sublist in totalslice]
                totalsslice = [[sublist[i] for i in self.kms_[dim].ordidx] for sublist in totalsslice]
                totalbpoints = [[sublist[i] for i in self.kms_[dim].ordidx] for sublist in totalbpoints]
                bbox_preds = [[sublist[i] for i in self.kms_[dim].ordidx] for sublist in bbox_preds]


                ############# post-hoc filtering
            totalslice = [item for sublist in totalslice for item in sublist]
            totalsslice = [item for sublist in totalsslice for item in
                           sublist]  # this can be treated as weights in samples
            bbox_preds = np.concatenate(bbox_preds)

            self.totalbpoints = totalbpoints  # DELLLLLLLLLLLLL

            if refit and self.cluster_override_factor[self.feature_names[dim]] is not None:
                # Update number of clusters based on the refined breakpoints
                sum_bkps = sum([len(bp) for bp in totalbpoints])
                self.n_clusters[self.feature_names[dim]] = max(2, int(sum_bkps / X.shape[0] *
                                                                      self.cluster_override_factor[
                                                                          self.feature_names[dim]] + 0.5))

            if self.verbose > 0:
                end_time = time.time()
                print(f'Done in {(end_time - start_time)}.')
                start_time = time.time()
                print(f'Clustering data')
            X_bis_o = to_time_series_dataset(totalslice)
            X_bis_o_shap = to_time_series_dataset(totalsslice)


            if not refit:
                if self.method not in ['dtw', 'rocket', 'sbc']:  # FX4
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

            if self.method in ['tskshape', 'kmeans', 'shapelet']:  # FX5
                # X_bis = np.nan_to_num(X_bis) #TODO: or
                from sklearn.impute import SimpleImputer
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                X_bis[:, :, 0] = imp.fit_transform(X_bis[:, :, 0])
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
                if self.verbose > 0:
                    start_time = time.time()
                    print(f'Clustering data')
                if self.method == 'dtw':
                    self.kms_[dim] = TimeSeriesKMeans(n_clusters=self.n_clusters[self.feature_names[dim]],
                                                      metric="dtw", max_iter=5,
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
                elif self.method == 'rocket':
                    self.kms_[dim] = RocketMedoids(n_clusters=self.n_clusters[self.feature_names[dim]],
                                                   n_jobs=self.n_jobs)  # FX4
                elif self.method == 'shapelet':  # FX6
                    # FX99
                    non_nan_counts = np.sum(~np.isnan(X_bis_o), axis=1)
                    min_shapelet_length = self.min_size
                    max_shapelet_length = min(max(non_nan_counts), X_bis.shape[1])[0]
                    self.kms_[dim] = ShapeletClustering(max_shapelets=self.n_clusters[self.feature_names[dim]],
                                                        min_shapelet_length=min_shapelet_length,
                                                        max_shapelet_length=max_shapelet_length)
                elif self.method == 'sbc':
                    self.kms_[dim] = SequentialBreakpointClustering(ordidx)

                self.seq_len_[dim] = X_bis.shape[1]

                self.kms_[dim].fit(X_bis, y=bbox_preds)


                if self.method == 'kshape':
                    self.kms_[dim].cluster_centers_ = self.kms_[dim].centroids_
                if self.method == 'kshapegpu':
                    self.kms_[dim].cluster_centers_ = self.kms_[dim].centroids_.detach().cpu()

            self.xbis_cluster_labels_[self.feature_names[dim]] = self.kms_[dim].labels_  # FX2
            self.xbis_ordidx_[self.feature_names[dim]] = ordidx
            # FX99 - move refit

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

                self.xbis_cluster_labels_[self.feature_names[dim]] = labels  # FX2
                self.signal_ids_[self.feature_names[dim]] = np.concatenate(indexes)  # FX2
                # todo xbis and cluster centers should be changed here?

            # FX99 return to the shapes used by the rest of tsproto
            if self.method == 'kmeans':
                X_bis = X_bis.reshape(X_bis.shape[0], X_bis.shape[1], 1)
                X_bis_shap = X_bis_shap.reshape(X_bis_shap.shape[0], X_bis_shap.shape[1], 1)
            self.xbis_[self.feature_names[dim]] = X_bis  # FX2
            self.xbis_shap_[self.feature_names[dim]] = X_bis_shap  # FX2

            if not transform:
                return self

            Xdf['cluster'] = labels
            Xdf['shapweight'] = np.array([self.importance_aggregation_func(abs(sv)) for sv in totalsslice])
            Xdf['startpoint'] = np.concatenate(totalbpoints)  # FX
            Xdf['durations'] = X_bis_o.shape[1] - np.sum(np.isnan(X_bis), axis=1)

            Xdf['min'] = np.nanmin(X_bis, axis=1)
            Xdf['max'] = np.nanmax(X_bis, axis=1)
            Xdf['mean'] = np.nanmean(X_bis, axis=1)
            Xdf['std'] = np.nanstd(X_bis, axis=1)
            Xdf['trend'] = calculate_trends(X_bis)  # FX3
            Xdf['frequency'] = dominant_frequencies_for_rows(X_bis, sampling_rate=self.sampling_rate_)

            phantom = pd.DataFrame({'sigid': [-1] * len(self.label_features_[dim]),
                                    'cluster': np.arange(0, len(self.label_features_[dim])),
                                    'startpoint': np.arange(0, len(self.label_features_[dim])),
                                    'durations': [0] * len(self.label_features_[dim]),
                                    'min': [0] * len(self.label_features_[dim]),
                                    'max': [0] * len(self.label_features_[dim]),
                                    'mean': [0] * len(self.label_features_[dim]),
                                    'std': [0] * len(self.label_features_[dim]),
                                    'trend': [0] * len(self.label_features_[dim]),
                                    'frequency': [0] * len(self.label_features_[dim])
                                    })

            Xdfp = pd.concat((Xdf, phantom))

            # TODO: this works only for dtw & shapelet
            if self.method in ['dtw', 'shapelet']:
                distances_to_barycenters = self.kms_[dim].transform(X_bis)
            elif self.method in ['kmeans']:
                # Kmeans uses 2D arrays
                distances_to_barycenters = self.kms_[dim].transform(X_bis.reshape(X_bis.shape[0], X_bis.shape[1]))
            elif self.method in ['kshape', 'tskshape', 'kshapegpu', 'rocket', 'sbc']:
                barycenters = self.kms_[dim].cluster_centers_
                # Compute distances from each time series to each barycenter using cdist with ccor metric
                distances_to_barycenters = cdist_kshape(X_bis, barycenters)
                # Find the closest barycenter for each time series
                # distances_to_barycenters = np.argmin(distances, axis=1)

            if isinstance(distances_to_barycenters, np.ndarray):
                dsts = pd.DataFrame(distances_to_barycenters)  # FX6
            else:  # FX6
                dsts = distances_to_barycenters  # FX6

            dsts.columns = [f'distance_cl_{c}_{self.feature_names[dim]}' for c in range(0, dsts.shape[
                1])]

            dsts.loc[:, 'sigid'] = Xdf['sigid'].values  # FX6
            ohe_dst = dsts.groupby('sigid').min()  # FX6


            ohe_train = pd.pivot_table(Xdfp, index='sigid', columns='cluster', values='durations',
                                       aggfunc=lambda x: sum(~np.isnan(x))).fillna(0).astype(int)

            startpoint_train = pd.pivot_table(Xdfp, index='sigid', columns='cluster', values='startpoint',
                                              aggfunc='min').fillna(0)
            duration_train = pd.pivot_table(Xdfp, index='sigid', values='durations',
                                            columns='cluster').fillna(0)
            min_train = pd.pivot_table(Xdfp, index='sigid', values='min', columns='cluster',
                                       aggfunc='min').fillna(min_float32)
            max_train = pd.pivot_table(Xdfp, index='sigid', values='max', columns='cluster',
                                       aggfunc='max').fillna(max_float32)
            mean_train = pd.pivot_table(Xdfp, index='sigid', values='mean', columns='cluster').fillna(0)
            std_train = pd.pivot_table(Xdfp, index='sigid', values='std', columns='cluster').fillna(0)
            trend_train = pd.pivot_table(Xdfp, index='sigid', values='trend', columns='cluster').fillna(0)
            frequency_train = pd.pivot_table(Xdfp, index='sigid', values='frequency',
                                             columns='cluster').fillna(0)
            self.weights_[dim] = Xdf.groupby('sigid')['shapweight'].apply(self.importance_aggregation_func).values


            ohe_train.columns = [f'exists_cl_{c}_{self.feature_names[dim]}' for c in
                                 ohe_train.columns]
            duration_train.columns = [f'duration_cl_{c}_{self.feature_names[dim]}' for c in
                                      duration_train.columns]
            min_train.columns = [f'min_cl_{c}_{self.feature_names[dim]}' for c in
                                 min_train.columns]
            max_train.columns = [f'max_cl_{c}_{self.feature_names[dim]}' for c in
                                 max_train.columns]
            mean_train.columns = [f'mean_cl_{c}_{self.feature_names[dim]}' for c in
                                  mean_train.columns]
            std_train.columns = [f'std_cl_{c}_{self.feature_names[dim]}' for c in
                                 std_train.columns]
            trend_train.columns = [f'trend_cl_{c}_{self.feature_names[dim]}' for c in
                                   trend_train.columns]
            frequency_train.columns = [f'frequency_cl_{c}_{self.feature_names[dim]}' for c in
                                       frequency_train.columns]
            startpoint_train.columns = [f'startpoint_cl_{c}_{self.feature_names[dim]}' for c in
                                        startpoint_train.columns]

            train = pd.concat((ohe_train, duration_train, min_train, max_train, mean_train, std_train, trend_train,
                               frequency_train, startpoint_train, ohe_dst), axis=1)  # FX6
            train = train[~train.index.isin([-1])]

            print(self.descriptors)

            if 'existance' in self.descriptors and self.method not in ['sbc']:
                features = features + list(ohe_train.columns)
            if 'similarity' in self.descriptors and self.method not in ['sbc']:
                features = features + list(ohe_dst.columns)
            if 'duration' in self.descriptors and self.method not in ['sbc']:
                features = features + list(duration_train.columns) + list(startpoint_train.columns)
            if 'stats' in self.descriptors:
                features = features + list(min_train.columns) + list(max_train.columns) + list(
                    mean_train.columns) + list(std_train.columns) + list(trend_train.columns)
            if 'frequency' in self.descriptors:
                features = features + list(frequency_train.columns)

            train_dim.append(train)

        train = pd.concat(train_dim, axis=1)
        target = 'target'

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


class SequentialBreakpointClustering(BaseEstimator, TransformerMixin):
    def __init__(self, ordidx):
        self.ordidx = np.sort(np.unique(ordidx))

    def fit(self, X, y=None):
        self.labels_ = np.tile(self.ordidx, int(len(X) / len(
            self.ordidx)))
        self.cluster_centers_ = np.array([X[self.labels_ == i].mean(axis=0) for i in range(np.max(self.labels_) + 1)
                                          ]).reshape(-1, X.shape[1], 1)
        return self

    def predict(self, X):
        if self.labels_ is None:
            raise ValueError("The model has not been fitted yet.")

        return np.tile(self.ordidx, int(len(X) / len(self.ordidx)))

class RocketMedoids:

    def __init__(self,n_clusters=2, n_kernels=512, n_components=150,n_jobs=1):
        self.rocket = MiniRocket(num_kernels=n_kernels,n_jobs=n_jobs)
        self.kmedoids = kmedoids.KMedoids(n_clusters=n_clusters,method='fasterpam')
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.pca_medoids_ = None
    def fit(self,X,y=None):
        X = X.reshape(X.shape[0],1,X.shape[1])
        Xr = self.rocket.fit_transform(X, y)
        Xr = Xr.replace([np.inf, -np.inf], np.nan).fillna(0)
        Xr=self.scaler.fit_transform(Xr)
        Xrd = self.pca.fit_transform(Xr)
        self.kmedoids.fit(euclidean_distances(Xrd))
        self.cluster_centers_ = X[self.kmedoids.medoid_indices_]#self.rocket.inverse_transform(self.pca.inverse_transform(self.kmedoids.cluster_centers_))
        self.pca_medoids_ = Xrd[self.kmedoids.medoid_indices_]
        self.labels_ = self.kmedoids.labels_
    def predict(self,X):
        X = X.reshape(X.shape[0],1,X.shape[1])
        dists = euclidean_distances(self.pca.transform(self.scaler.transform(self.rocket.transform(X).fillna(0))), self.pca_medoids_)
        return np.argmin(dists, axis=1)


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





class ShapeletClustering:
    def __init__(self, max_shapelets=None, min_shapelet_length=5, max_shapelet_length=30, random_state=None):
        self.shapelet_transform = RandomShapeletTransform(
            max_shapelets=max_shapelets,
            min_shapelet_length=min_shapelet_length,
            max_shapelet_length=max_shapelet_length,
            random_state=random_state
        )
        self.fitted = False
        self.shapelets_ = None

    def preprocess_(self, X):
        data_rearranged = np.transpose(X, (0, 2, 1))
        X_proc = from_3d_numpy_to_nested(data_rearranged)
        return X_proc

    def fit(self, X, y=None):
        # Fit the shapelet transform
        self.XX = X
        self.yy = y
        X = self.preprocess_(X)
        self.shapelet_transform.fit(X, y)
        self.fitted = True
        self.shapelets_ = self.shapelet_transform.shapelets
        self.cluster_centers_ = [s[6] for s in self.shapelets_]
        self.labels_ = self.predict(X, preprocess=False)

    def transform(self, X, preprocess=True):
        if not self.fitted:
            raise ValueError(
                "The ShapeletTransformer is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if preprocess:
            X = self.preprocess_(X)
        # Transform the input data using the fitted shapelet transform
        return self.shapelet_transform.transform(X)

    def predict(self, X, preprocess=True):
        if not self.fitted:
            raise ValueError(
                "The ShapeletTransformer is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        if preprocess:
            X = self.preprocess_(X)
        # Calculate distances to each shapelet and find the shapelet with the minimal distance
        transformed_X = self.transform(X, preprocess=False)
        shapelet_indices = np.argmin(transformed_X, axis=1)

        return shapelet_indices