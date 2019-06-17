# script for performing prediction on new data
# by Bing Zhao
# updated on 2019.06.17
# bing.zhao.bzh@gmail.com

from utf_conversion import *
from utf_aggregation import *
from utf_utilities import *
from utf_features import *

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# load interquantile range for outlier removal
dir_model = '../model/'
with open(dir_model+'df_IQR.pickle', 'rb') as handle:
    df_IQR = pickle.load(handle)
with open(dir_model+'df_IQR_zc.pickle', 'rb') as handle:
    df_IQR_zc = pickle.load(handle)
with open(dir_model+'scaler_standard.pickle', 'rb') as handle:
    scaler_standard = pickle.load(handle)
with open(dir_model+'imp_median.pickle', 'rb') as handle:
    imp_median = pickle.load(handle)
with open(dir_model+'clf_xgb.pickle', 'rb') as handle:
    clf_xgb = pickle.load(handle)

# read command line input
# to use: python src_baseline/read_command_line.py --input test.csv --output test.csv
parser = argparse.ArgumentParser(description='DengerDriverPrediction')
parser.add_argument('--input',default='../test/test.csv', help='path for the test csv')
parser.add_argument('--output',default='../test/my_prediction.csv', help='path for the output prediction results')
args = parser.parse_args()
file_in = args.input
file_out = args.output

# read raw data
df = pd.read_csv(file_in, low_memory=False)

# sort by bookingID and second
df.drop_duplicates(subset=None, keep='first', inplace=True)
df.sort_values(by=['bookingID', 'second'], inplace=True)
df.reset_index(inplace=True, drop=True)

# outlier removal
df = func_filter_outlier_iqr(df, df_IQR)

# feature engineering
df = func_df_mag_rms_orient_a(df, ['acceleration_x', 'acceleration_y', 'acceleration_z'])
df = func_df_mag_rms_orient_g(df, ['gyro_x', 'gyro_y', 'gyro_z'])

# construct Operations for Aggregation per BookingID
# simply Numpy Functions: Mean, Std, Var, Median, Max-Min
features_stats_cols = ['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y', 'acceleration_z',
                       'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed', 'a_rms', 'a_phi', 'a_theta',
                       'g_rms', 'g_phi', 'g_theta']
lb_min_max = lambda x: np.nanmax(x) - np.nanmin(x)
lb_min_max.__name__ = "range"
features_stats_functions = [np.nanmean, np.nanstd, np.nanvar, lb_min_max]
ops_stats = {col: features_stats_functions for col in features_stats_cols}

# zero crossing
lb_zero_crossing = lambda x: np.nansum((x.values[~np.isnan(x.values)][:-1] * x.values[~np.isnan(x.values)][1:]) < 0)
lb_zero_crossing.__name__ = "zero_crossing"
features_zero_crossing_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z']
ops_zc = {col: [lb_zero_crossing] for col in features_zero_crossing_cols}

# Peak to Average Ratio
lb_par = lambda x: np.nanmax(x.values) / (np.nanmean(x.values) + 1e-12)
lb_par.__name__ = "par"
features_par_cols = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z',
                     'Speed', 'a_rms', 'g_rms']
ops_par = {col: [lb_par] for col in features_par_cols}

# Signal Magnitude Vector
lb_smv_func = func_signal_mag_vector
lb_smv_func.__name__ = "smv"
ops_smv = {col: [lb_smv_func] for col in ['a_rms', 'g_rms']}

# FFT: Spectral Energy and Entropy
lb_fft_func = lambda x: func_fft_spectral_energy(x.values)
lb_fft_func.__name__ = "fft"
ops_fft = {col: [lb_fft_func] for col in ['acceleration_x', 'acceleration_y', 'acceleration_z', 'a_rms',
                                          'gyro_x', 'gyro_y', 'gyro_z', 'g_rms']}

# Hjorth: (Activity, Mobility, Complexity)
lb_hjorth_func = lambda x: func_hjorth_parameters(x.values)
lb_hjorth_func.__name__ = "hjorth"
ops_hjorth = {col: [lb_hjorth_func] for col in
              ['acceleration_x', 'acceleration_y', 'acceleration_z', 'a_rms', 'a_phi', 'a_theta',
               'gyro_x', 'gyro_y', 'gyro_z', 'g_rms', 'g_phi', 'g_theta']}

# lambda function: calculate portion of missing values
features_stats_cols = ['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y', 'acceleration_z',
                       'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']
lb_nan_perc = lambda x: np.sum(np.isnan(x)) / len(x)
lb_nan_perc.__name__ = "nan_perc"
features_stats_functions = [lb_nan_perc]
ops_miss = {col: features_stats_functions for col in features_stats_cols}

# construct all the ops
ops = {}
ops_list = [
    ops_stats,
    ops_zc,
    ops_par,
    ops_smv,
    ops_fft,
    ops_hjorth,
    ops_miss]
for op in ops_list:
    for name, function_list in op.items():
        if name in ops.keys():
            ops[name].extend(function_list)
        else:
            ops.update({name: function_list.copy()})  # make sure using .copy(), otherwise, list appending dynamically

# aggregation per bookingID
gp = df.groupby('bookingID')
data_agg = gp.agg(ops)
data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns.values]

# convert tuple into columns
# (Activity, Mobility, Complexity)
for name in ops_hjorth.keys():
    old_name = name+"_hjorth"
    new_name1 = old_name + "_act"
    new_name2 = old_name + "_mob"
    new_name3 = old_name + "_com"
    data_agg[[new_name1, new_name2, new_name3]] = pd.DataFrame(data_agg[old_name].tolist(), index=data_agg.index)
    data_agg.drop(old_name, axis =1,  inplace=True)

# FFT: Spectral Energy and Entropy
for name in ops_fft.keys():
    old_name = name+"_fft"
    new_name1 = old_name + "_energy"
    new_name2 = old_name + "_entropy"
    data_agg[[new_name1, new_name2]] = pd.DataFrame(data_agg[old_name].tolist(), index=data_agg.index)
    data_agg.drop(old_name, axis =1,  inplace=True)
data_agg_dist = gp.apply(funct_time_integral, ['Speed'])
data_agg_cc_a = gp.apply(funct_inter_cross_correlation, ['acceleration_x', 'acceleration_y', 'acceleration_z'])
data_agg_cc_g = gp.apply(funct_inter_cross_correlation, ['gyro_x', 'gyro_y', 'gyro_z'])
data_agg_sma = gp.apply(funct_sma, ['acceleration_x', 'acceleration_y', 'acceleration_z', 'a_rms',
                                    'gyro_x', 'gyro_y', 'gyro_z', 'g_rms'])
data_agg_svm = gp.apply(funct_dsvm, ['a_rms', 'g_rms'])

# combine single and multiple features
df_X = pd.merge(data_agg, data_agg_dist, how = "left", on="bookingID")
df_X = pd.merge(df_X, data_agg_cc_a, how = "left", on="bookingID")
df_X = pd.merge(df_X, data_agg_cc_g, how = "left", on="bookingID")
df_X = pd.merge(df_X, data_agg_sma, how = "left", on="bookingID")
df_X = pd.merge(df_X, data_agg_svm, how = "left", on="bookingID")

# zero crossing features
col_zc = ['acceleration_x_zero_crossing', 'acceleration_y_zero_crossing', 'acceleration_z_zero_crossing', 'gyro_x_zero_crossing', 'gyro_y_zero_crossing', 'gyro_z_zero_crossing']
col_zc_perTD = []
for c in col_zc:
    c_t = c + 'perT'
    c_d = c + 'perD'
    df_X[c_t] = df_X[c] / (df_X['second_range'] / 60.0 + 1e-12)  # per minute
    df_X[c_d] = df_X[c] / (df_X['Speed_TimeIntegral'] + 1e-12)  # per kilometer, Speed_TimeIntegral = Trip Distance
    col_zc_perTD.append(c_t)
    col_zc_perTD.append(c_d)

# remove outliers
df_X[col_zc_perTD] = func_filter_outlier_iqr(df_X[col_zc_perTD], df_IQR_zc)

# Scale and impute numerical features
df_X_test = scaler_standard.transform(df_X)
df_X_test = imp_median.transform(df_X_test)

# prediction_period_multiplier
y_pred_test = clf_xgb.predict(df_X_test)
y_pred_test_proba = clf_xgb.predict_proba(df_X_test)

# construct final results
df_prediction = pd.DataFrame({'bookingID': df_X.index.values, 'prediction': y_pred_test, 'prediction_proba': y_pred_test_proba[:,1]})
df_prediction.to_csv(file_out, index=False)
