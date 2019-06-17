import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas import DataFrame
from utf_conversion import *
from utf_aggregation import *
from utf_utilities import *

#TODO: Outlier Removal



def func_df_mag_rms_orient_a(df, xyz_names):
	"""function to calculate magnitude, rms and orientations
	Keyword arguments: df (dataframe); Return: df_summary
	"""
	a_mag, a_rms = func_mag_rms(df[xyz_names[0]].values, df[xyz_names[1]].values, df[xyz_names[2]].values)
	a_phi, a_theta = func_orientation_angles(df[xyz_names[0]].values, df[xyz_names[1]].values, df[xyz_names[2]].values)
	df = df.assign(a_mag=a_mag, a_rms=a_rms, a_phi=a_phi, a_theta=a_theta)
	return df


def func_df_mag_rms_orient_g(df, xyz_names):
	"""function to calculate magnitude, rms and orientations
	Keyword arguments: df (dataframe); Return: df_summary
	"""
	a_mag, a_rms = func_mag_rms(df[xyz_names[0]].values, df[xyz_names[1]].values, df[xyz_names[2]].values)
	a_phi, a_theta = func_orientation_angles(df[xyz_names[0]].values, df[xyz_names[1]].values, df[xyz_names[2]].values)
	df = df.assign(g_mag=a_mag, g_rms=a_rms, g_phi=a_phi, g_theta=a_theta)
	return df


def funct_time_integral(df, labels):
	"""function to calculate time integral
	Keyword arguments: df (dataframe); Return: df_summary
	"""
	dics = {}
	for v in labels:
		name = "{}_TimeIntegral".format(v)
		value = func_trip_distance(df.second.values, df[v].values)
		dics.update({name: value})
	return pd.Series(dics, index=dics.keys())


def funct_inter_cross_correlation(df: DataFrame, labels: list) -> pd.Series:
	dics = {}
	for i in range(len(labels) - 1):
		for j in range(i + 1, len(labels)):
			x, y = df[labels[i]].values, df[labels[j]].values
			na_mask = np.isnan(x) | np.isnan(y)
			value = np.correlate(x[~na_mask], y[~na_mask])[0]
			name = "{}_{}_cc".format(labels[i], labels[j])
			dics.update({name: value})
	return pd.Series(dics, index=dics.keys())


# Signal Magnitude Area
def funct_sma(df, labels):
	dics = {}
	for v in labels:
		name = "{}_sma".format(v)
		value = func_signal_mag_area(df.second.values, df[v].values)
		dics.update({name: value})
	return pd.Series(dics, index=dics.keys())


# Differential Signal Vector Magnitude
def funct_dsvm(df, labels):
	dics = {}
	for v in labels:
		name = "{}_svm".format(v)
		value = func_diff_signal_vector_mag(df.second.values, df[v].values)
		dics.update({name: value})
	return pd.Series(dics, index=dics.keys())


def func_processing(df0_test, df_IQR, df_IQR_zc, ops, ops_hjorth, ops_fft):
	# outlier removal
	df0_test = func_filter_outlier_iqr(df0_test, df_IQR)
	# feature engineering
	df0_test = func_df_mag_rms_orient_a(df0_test, ['acceleration_x', 'acceleration_y', 'acceleration_z'])
	df0_test = func_df_mag_rms_orient_g(df0_test, ['gyro_x', 'gyro_y', 'gyro_z'])
	# aggregation per bookingID
	gp = df0_test.groupby('bookingID')
	data_agg = gp.agg(ops)
	data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns.values]
	# convert tuple into columns
	#(Activity, Mobility, Complexity)
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

	return df_X
