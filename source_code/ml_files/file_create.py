import json
import pandas as pd
from feature_select import *  


def partition_locations(feature_score_file, locations):
    f = open(feature_score_file)
    data = json.load(f)
    for l in locations:
            with open('data/nyiso/locations/features_{}.txt'.format(l), "w") as outfile:
                outfile.write('')

    for i in data.keys():
        for l in locations:
            if i.find(l)!=-1:
                with open('data/nyiso/locations/features_{}.txt'.format(l), "a") as outfile:
                    outfile.write('{}: {}\n'.format(i, data[i]))
    f.close()

def create_one_hot_feature(params, feature_file_name, categorical_columns):
    feature_name = pd.read_csv('{}.csv'.format(feature_file_name))
    for column in categorical_columns:
        tempdf = pd.get_dummies(feature_name[column], prefix=column)
        feature_name = pd.merge(
            left=feature_name,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        feature_name = feature_name.drop(columns=column)
    feature_name.to_csv("{}.csv".format(feature_file_name), index=False)

def create_reduced_feature_file(feature_file_name, feature_score_file, categorical_columns_after_one_hot, threshold):
    f = open(feature_score_file)
    feature_scores = json.load(f)
    feature_file_data = pd.read_csv('{}.csv'.format(feature_file_name))
    feature_file_data.drop(columns = [feature_name for feature_name in feature_scores.keys() \
        if feature_name not in categorical_columns_after_one_hot and feature_scores[feature_name]<threshold],\
            inplace=True)
    feature_file_data.to_csv("{}_red.csv".format(feature_file_name), index=False)

def prepare_feature_selection_data(params, feature_file_name):
    ts_data = pd.read_csv(f'{feature_file_name}.csv', index_col = 0, parse_dates=['Datetime'])
    target_data = pd.read_csv(f'data/{params["system"]}/target_data_feature_selection.csv', index_col = 0, parse_dates=['Datetime'])
    return ts_data, target_data

## Start feature selection

def feature_selection_model(rfe_feature_number, n_train_days, categorical_columns, \
        categorical_columns_after_one_hot, feature_score_file, threshold):
    with open("params.json") as json_file:
        params = json.load(json_file)
    feature_file_name = f'data/{params["system"]}/ts_data_feature_selection' 
    feature_score_file = f'data/{params["system"]}/' + feature_score_file
    # create_one_hot_feature(params, feature_file_name, categorical_columns)
    ts_data, target_data = prepare_feature_selection_data(params, feature_file_name)
    select_features("first stage", params, ts_data, target_data, feature_file_name, rfe_feature_number, n_train_days)
    create_reduced_feature_file(feature_file_name, feature_score_file, categorical_columns_after_one_hot, threshold)
    feature_file_name = f'data/{params["system"]}/ts_data_feature_selection_red'
    ts_data, target_data = prepare_feature_selection_data(params, feature_file_name)
    select_features("second stage", params, ts_data, target_data, feature_file_name, rfe_feature_number, n_train_days)

rfe_feature_number = 66
n_train_days = 730
categorical_columns = ['is_weekend']
categorical_columns_after_one_hot = ['is_weekend_0', 'is_weekend_1']
threshold = 9 # 24 * 5 * 0.6
feature_score_file = "mi_features.json"
# Number of columns in the original dataset: 1326 + Datetime
feature_selection_model(rfe_feature_number, n_train_days, categorical_columns, \
    categorical_columns_after_one_hot, feature_score_file, threshold)

