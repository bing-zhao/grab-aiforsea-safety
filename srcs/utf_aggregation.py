import numpy as np
import pandas as pd


# function to display agg results
def funct_agg_display(gp, cop):
    """
    function to display aggregation results
    :param gp: group object returned by df.groupby
    :param cop: dinctionary containing operation fnctions
    :return: display the head of resulting dataframe
    """
    features_stats = gp.agg(cop)
    features_stats.columns = ['_'.join(col).strip() for col in features_stats.columns.values]
    return features_stats.head().T
    pass


def func_trip_distance(t, a):
    """
    function to calculate Differential Signal Vector Magnitude (DSVM)
    :param t: time vector (sorted by ascending order) (numpy array)
    :param a: speed (numpy array)
    :return: dist
    """
    mask = ~(np.isnan(t) | np.isnan(a))
    t = t[mask]
    a = a[mask]
    if len(a) > 1:
        del_t = np.diff(t)
        a_avg = (a[:-1] + a[1:]) / 2.0
        dist = np.nansum(del_t * a_avg)
    else:
        dist = np.nan
    return dist


def func_signal_mag_area(t, a):
    """
    function to calculate Signal Magnitude Area (SMA)
    :param t: time vector (sorted by ascending order) (numpy array)
    :param a: corresponding signal (numpy array)
    :return: sma
    """
    mask = ~(np.isnan(t) | np.isnan(a))
    t = t[mask]
    a = a[mask]
    if len(a) > 1:
        del_t = np.diff(t)
        avg_a = (np.abs(a[:-1]) + np.abs(a[1:]))/2.0
        T = np.nanmax(t) - np.nanmin(t)
        sma = np.nansum(avg_a * del_t) / T
    else:
        sma = np.nan
    return sma


def func_signal_mag_vector(a):
    """
    function to calculate Signal Magnitude Vector (SMV)
    :param a: signal (numpy array)
    :return: smv
    """
    sma = np.sqrt(np.nansum(np.power(a, 2))) / len(a)
    return sma


def func_diff_signal_vector_mag(t, a):
    """
    function to calculate Differential Signal Vector Magnitude (DSVM)
    :param t: time vector (sorted by ascending order) (numpy array)
    :param a: corresponding signal (numpy array)
    :return: dsvm
    """
    mask = ~(np.isnan(t) | np.isnan(a))
    t = t[mask]
    a = a[mask]
    if len(a) > 2:
        k = 2
        del_t = np.diff(t)[k-1:]
        a_dk = np.abs(a[k:] - a[k-1:-1])
        a_dk_minus1 = np.abs(a[k-1:-1] - a[k-2:-2])
        avg_a_dk = (a_dk_minus1 + a_dk)/2.0
        T = np.nanmax(t) - np.nanmin(t)
        dsvm = np.nansum(avg_a_dk * del_t) / T
    else:
        dsvm = np.nan
    return dsvm


def func_fft_spectral_energy(a):
    """
    function to perform FFT analysis and calculate spectral energy and entropy
    :param a: signal (numpy array)
    :return: E: energy, H: entropy
    """
    mask = ~np.isnan(a)
    a = a[mask]

    if len(a) > 1:
        a_fft = np.fft.fft(a)
        # power spectral density
        a_psd = np.abs(a_fft)**2 / len(a_fft)
        # power spectral energy
        E = np.nansum(a_psd)
        # normalized psd
        a_psdn = a_psd / (np.sum(a_psd) + 1e-12)
        # Entropy
        H = -np.nansum(a_psdn * np.log(a_psdn))
    else:
        E = np.nan
        H = np.nan
    return E, H


def func_diff_shift_k(b, k):
    """
    function to calculate the difference between
    :param b: numpy array
    :param k: interval, i.e. bk - b0
    :return: numpy array of diff
    """
    return (b[k:] - b[:-k])


def func_hjorth_parameters(a):
    """
    function to calculate Hjorth Parameters
    :param a: signal (numpy array)
    :return: Activity, Mobility, Complexity
    """
    mask = ~np.isnan(a)
    a = a[mask]
    if len(a) > 3:
        # Activity
        d0 = func_diff_shift_k(a, 1)
        A = np.nansum(d0 ** 2) / np.sum(~np.isnan(d0))

        # Mobility
        d1 = func_diff_shift_k(d0, 1)
        m1 = np.nansum(d1 ** 2) / np.sum(~np.isnan(d1))
        M = np.sqrt(m1 / (A + 1e-12))

        # Complexity
        d2 = func_diff_shift_k(d1, 1)
        m2 = np.nansum(d2 ** 2) / np.sum(~np.isnan(d2))
        C = np.sqrt(m2 / (m1 + 1e-12))
    elif len(a) > 2:
        # Activity
        d0 = func_diff_shift_k(a, 1)
        A = np.nansum(d0 ** 2) / np.sum(~np.isnan(d0))

        # Mobility
        d1 = func_diff_shift_k(d0, 1)
        m1 = np.nansum(d1 ** 2) / np.sum(~np.isnan(d1))
        M = np.sqrt(m1 / (A + 1e-12))

        # Complexity
        C = np.nan
    elif len(a) > 1:
        # Activity
        d0 = func_diff_shift_k(a, 1)
        A = np.nansum(d0 ** 2) / np.sum(~np.isnan(d0))
        # Mobility
        M = np.nan
        # Complexity
        C = np.nan
    else:
        # Activity
        A = np.nan
        # Mobility
        M = np.nan
        # Complexity
        C = np.nan

    return A, M, C
