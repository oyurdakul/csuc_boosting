from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import datetime
from datetime import date
import csv
import time
import os
import json
import shutil, errno
from matplotlib import pyplot
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

def select_features_f(X_train, y_train):
	# configure to select all features
	fs= SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)
	fs.transform(X_train)
	return fs

def select_features_MI(X_train, y_train):
	# configure to select all features
	fs= SelectKBest(score_func=mutual_info_regression, k='all').fit(X_train, y_train)
	fs.transform(X_train)
	return fs

def select_features_anova(X_train, y_train):
	# configure to select all features
	fs= SelectKBest(score_func=f_classif, k='all').fit(X_train, y_train)
	fs.transform(X_train)
	return fs

def select_features_p(X_train, y_train):
	# configure to select all features
	p_vals= SelectKBest(score_func=r_regression, k='all').fit(X_train, y_train)
	p_vals.transform(X_train)
	return p_vals

def select_features_rfe(X_train, y_train, feat_num, feature_names):
	rf = RandomForestRegressor(n_estimators=100,
								max_depth=6,
								max_features=0.4)
	rfe = RFE(estimator=rf, n_features_to_select=feat_num, step = 1)
	rfe = rfe.fit(X_train, y_train)
	return rfe.get_feature_names_out(feature_names)

def rfe_experiment(feature_file, feat_num, X_train, y_train, feature_names):
	maintained = select_features_rfe(X_train, y_train, feat_num, feature_names)
	feature_name = pd.read_csv('{}.csv'.format(feature_file))
	print(maintained)
	for i in feature_name.columns:
		if i not in maintained:
			if i!='Datetime':
				feature_name.drop(i, inplace=True, axis=1)
	feature_name.to_csv("{}_rfe.csv".format(feature_file), index=False)


def f_experiment(X_train, y_train, system, feature_names, output_dim):
	fs = select_features_f(X_train, y_train[:,0])
	p_values = np.nan_to_num(fs.pvalues_)
	f_scores = np.nan_to_num(fs.scores_)
	for i in range(1,output_dim):
		p_values += abs(np.nan_to_num(select_features_f(X_train, y_train[:,i]).pvalues_))
		f_scores += (np.nan_to_num(select_features_f(X_train, y_train[:,i]).scores_.replace(np.nan, 0)))

	index = np.argsort(p_values, axis=-1, kind='quicksort', order=None)
	dict = {}

	for i in reversed(index):
		dict[feature_names[i]] = p_values[i]

	with open("data/{}/p_vals_for_f.json".format(system), "w") as outfile:
		json.dump(dict, outfile)

	index2 = np.argsort(f_scores, axis=-1, kind='quicksort', order=None)
	dict2 = {}

	for i in reversed(index2):
		if np.isnan(f_scores[i]):
			dict2[feature_names[i]] = 0
		else:
			dict2[feature_names[i]] = f_scores[i]

	with open("data/{}/f_features.json".format(system), "w") as outfile:
		json.dump(dict2, outfile)

def p_experiment(X_train, y_train, system, feature_names, output_dim):
	p_s = select_features_p(X_train, y_train[:,0])
	p_scores = np.nan_to_num(p_s.scores_)
	for i in range(1,output_dim):
		p_scores += (np.nan_to_num(select_features_p(X_train, y_train[:,i]).scores_))
	

	index_p = np.argsort(p_scores, axis=-1, kind='quicksort', order=None)
	dict_p = {}

	for i in reversed(index_p):
		dict_p[feature_names[i]] = p_scores[i]

	with open("data/{}/p_scores.json".format(system), "w") as outfile:
		json.dump(dict_p, outfile)

	abs_p_scores = abs(p_scores)
	index_abs_p = np.argsort(abs_p_scores, axis=-1, kind='quicksort', order=None)
	dict_abs_p = {}
	for i in reversed(index_abs_p):
		dict_abs_p[feature_names[i]] = abs_p_scores[i]

	with open("data/{}/abs_p_scores.json".format(system), "w") as outfile:
		json.dump(dict_abs_p, outfile)

def mi_experiment(X_train, y_train, system, feature_names, output_dim):
	mi_s = select_features_MI(X_train, y_train[:,0])
	mi_scores = mi_s.scores_
	for i in range(1,output_dim):
		mi_scores += (select_features_MI(X_train, y_train[:,i]).scores_)

	index3 = np.argsort(mi_scores, axis=-1, kind='quicksort', order=None)
	dict3 = {}

	for i in reversed(index3):
		dict3[feature_names[i]] = mi_scores[i]

	with open("data/{}/mi_features.json".format(system), "w") as outfile:
	    json.dump(dict3, outfile)

def select_features(stage, params, ts_data, target_data, feature_file_name, rfe_feature_number, n_train_days):
	start_time = time.time()
	if params["system"] == 'caiso':
		#max load in the caiso dataset: 41330 MW
		#peak load of the 14-bus system: 321.29 MW
		#total generation capacity of the 14-bus system: 765.31 MW
		load_scal = 765.31*0.80/41330
		output_dim = 24
	elif params["system"] == "nyiso":
		#max load in the nyiso dataset: 31866.4167 MW on Aug 29, 2018 at 5 pm
		load_scal = 765.31*0.80/31866.74
		# load_scal = 1
		# output_dim = 24 * 11
		output_dim = 24 * 5
	start_train = ts_data.index.get_loc(params["start_date_train"]) 
	end_train = start_train + n_train_days
	target_data = target_data * load_scal
	X_train = ts_data.iloc[start_train:end_train].values
	y_train = target_data.iloc[start_train:end_train].values
	X_scaler = MinMaxScaler()
	y_scaler = MinMaxScaler()
	X_scaler = X_scaler.fit(X_train)
	y_scaler = y_scaler.fit(y_train)
	X_train = X_scaler.transform(X_train)
	y_train = y_scaler.transform(y_train)
	feature_names = ts_data.columns
	if stage == "first stage":
		mi_experiment(X_train, y_train, params["system"], feature_names, output_dim)
	elif stage == "second stage":
		rfe_experiment(feature_file_name, rfe_feature_number, X_train, y_train, ts_data.columns)
	end_time = time.time()
	print(f'{stage} feature selection completed in {np.round(end_time - start_time, 3)} s')
	
