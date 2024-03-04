import numpy as np
import shap

from tsproto.windowshap import SlidingWindowSHAP

"""
Source code taken from: https://github.com/vsubbian/WindowSHAP
"""
def outliers(data, multiplier=1.5):
    """

    :param data:
    :param multiplier:
    :return:
    """
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



def getshap(model, X, y, shap_version='deep', bg_size=1000, stride=10, window_len=10, absshap=True, shuffle=True):
    indexes = np.arange(0, len(X))
    if shuffle:
        np.random.shuffle(indexes)
    maxid = min(bg_size, len(X))

    background_data = X[indexes[:maxid]]
    if shap_version == 'window':
        sv_tr = np.zeros((len(X), X.shape[1], X.shape[2]))

        for i in range(len(X)):
            gtw = SlidingWindowSHAP(model, stride, window_len, background_data, X[i:i + 1], model_type='lstm')
            sv_tr[i, :, :] = gtw.shap_values(num_output=y.shape[1])
    elif shap_version == 'deep':
        explainer = shap.DeepExplainer(model, background_data)
        shap_values_tr = explainer.shap_values(X, check_additivity=False)
        if absshap:
            sv_tr = abs(np.array(shap_values_tr)).mean(
                axis=0)  # This basically returns the average importance over the feature/sample
            # Not taking into account the sign of shap value, as it is not required
            # for breakpoints calculation
        else:
            indexer = np.argmax(model.predict(X), axis=1)
            sv_tr = []
            for i in range(0, len(X)):
                sv_tr.append([shap_values_tr[indexer[i]][i, :]])
            sv_tr = np.concatenate(sv_tr)
        return background_data,sv_tr