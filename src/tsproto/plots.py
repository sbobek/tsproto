import numpy as np
import pandas as pd
from graphviz import Digraph
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle
from ruptures.utils import pairwise
from tsproto.utils import dominant_frequencies_for_rows

COLOR_CYCLE = ["#4286f4", "#f44174"]


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
    elif stat_function == 'strartpoint':
        cdata = X_train[cluster_labels == cluster_index]
        cluster_values = np.nanmin(X_train[cluster_labels == cluster_index], axis=1)
        stat_name = 'startpoint'
    else:
        raise ValueError(
            "Invalid stat_function. Choose from 'max', 'min', 'std', 'mean', 'duration' or startpoint.")

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
                                   sampling_rate=proto_encoder.sampling_rate, threshold=threshold)

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




def display_breakpoints(
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


from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_smooth_colored_line(x_values, y_values, color_values, resolution=1000, ax=None, add_cbar=True):
    # Interpolate y-values and color values for the higher-resolution x-values
    x_values_high_res = np.linspace(x_values.min(), x_values.max(), resolution)
    y_values_interp = np.interp(x_values_high_res, x_values, y_values)
    color_values_interp = np.interp(x_values_high_res, x_values, color_values)

    # Normalize color values to be in the range [0, 1] for colormap mapping
    norm_color = Normalize(color_values_interp.min(), color_values_interp.max())
    colors = plt.cm.viridis(norm_color(color_values_interp))

    # Create a LineCollection with smooth gradient colors
    points = np.column_stack([x_values_high_res, y_values_interp])
    segments = np.column_stack([points[:-1], points[1:]])

    # Reshape segments to have a shape of (M, 2, 2)
    segments = segments.reshape(-1, 2, 2)

    lc = LineCollection(segments, cmap='viridis', norm=norm_color)
    lc.set_array(color_values_interp)

    # Plot the line collection
    if ax is None:
        fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    #ax.set_xlim(0, len(x_values) - 1)

    # Add colorbar
    if add_cbar:
        cbar = plt.colorbar(lc, ax=ax, label='SHAP Values')
    return ax