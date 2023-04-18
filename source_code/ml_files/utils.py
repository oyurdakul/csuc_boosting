from pydoc import doc
from attr import NOTHING
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor, train
from adaboost import AdaBoostR2
import pandas as pd
import numpy as np
import math
from datetime import date
import datetime
import csv
import time
import os
import json
import shutil, errno
from pathlib import Path
# from gbdtmo import GBDTMulti, load_lib, create_graph

# Plots
# ==============================================================================
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.colors import LinearSegmentedColormap
W = 5.8 
plt.rcParams.update({
    'figure.figsize': (W, W/(4/2.2)),     # 4:3 aspect ratio
    'font.size' : 11,                   # Set font size to 11pt
    'axes.labelsize': 11,               # -> axis labels
    'legend.fontsize': 11,              # -> legends
    'font.family': 'lmodern',
    'text.usetex': True,
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage{lmodern}'
        # ... more packages if needed
    )
}) 

def plot_scenario_weights(m, pol, hp, X_train_dates, INPUT_FILE_DIR, n_scenarios, oos_scenario):
    n_es = str(hp["n_estimators"])
    xi = str(hp["Xi"])
    m_d = str(hp["max_depth"])
    l_r = str(hp["learning_rate"])
    m_s_l = str(hp["min_split_loss"])
    m_s_s = str(hp["min_samples_split"])
    m_f = str(hp["max_features"])
    filename = INPUT_FILE_DIR + f'snum_{str(n_scenarios)}/oos_{str(oos_scenario)}/weighted/{str(m)}/n_est_{n_es}_Xi_{xi}_'\
            + f'max_depth_{m_d}_lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.png'
    fig, ax = plt.subplots()
    locator = mdate.AutoDateLocator()
    formatter = mdate.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.ylabel(r"weight $\omega_d(x)$")
    plt.xlabel(r"date of observation $d$")
    weights = pol.values()
    ax.plot(X_train_dates, weights, linestyle='solid', drawstyle = "steps-mid", linewidth=0.5, alpha=0.8, color = "salmon", marker='None')
    ax.fill_between(X_train_dates, weights, np.zeros(len(weights)), step="mid", alpha=0.25, color = "salmon")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close()

def prepare_forecast_data(ts_data,target_data, number_of_scenarios,load_scaling = 1, \
    number_of_oos_scenarios=1, start_date_train = False, start_date_oos = False):
    if start_date_train:
        start_train = ts_data.index.get_loc(start_date_train) 
    else:
        start_train = ts_data.index.get_loc('2018-05-15') 
    end_train = start_train + number_of_scenarios

    if start_date_oos:
        start_test = ts_data.index.get_loc(start_date_oos) 
    else:
        start_test = ts_data.index.get_loc('2019-07-15') 
    end_test = start_test + number_of_oos_scenarios
    target_data = target_data * load_scaling

    X_train = ts_data.iloc[start_train:end_train].values
    X_train_dates = ts_data.iloc[start_train:end_train].index.values
    y_train = target_data.iloc[start_train:end_train].values
    X_test  = ts_data.iloc[start_test:end_test].values
    y_test  = target_data.iloc[start_test:end_test].values

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaler = X_scaler.fit(X_train)
    y_scaler = y_scaler.fit(y_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    if len(X_train) != len(y_train):
        raise ValueError('Different feature and label sizes in the train set')
        
    if len(X_test) != len(y_test):
        raise ValueError('Different feature and label sizes in the test set')
    
    return X_train, X_test,y_train, y_test, X_scaler, y_scaler, X_train_dates

def tree_file_to_list(filename):
  with open(filename, "r") as f:
      lines = f.readlines()
      
  new_lines = []
  for l in lines:
    l = l.replace('\n','')
    l = l.replace('\t','')
    l = l.split(',')
    new_lines.append(l)

  booster_list = []
  temp = []
#   for l in new_lines:
#     print(f"{l[0]}")
  for l in  new_lines:
    if 'Booster' in l[0]:
      temp = [l[0]]
      booster_list.append(temp)
    else:
      temp.append(l)
  return booster_list

def booster_list_to_booster_dict(booster_list):
  booster = {}
  print(f"length of the booster list is {len(booster_list)}")
  for i,bs in enumerate(booster_list):
    nodes = {}
    for b_l in bs[1:]:
      node = {}
      if int(b_l[0]) < 0: #-> decision node
        node['node_type'] = 'decision'
        node['index'] = int(b_l[0])
        node['parent'] = int(b_l[1])
        node['left'] = int(b_l[2])
        node['right'] = int(b_l[3])
        node['split_col'] = int(b_l[4])
        node['split_val'] = float(b_l[5])
      if int(b_l[0]) > 0: #-> leaf node
        node['node_type'] = 'leaf'
        node['index'] = int(b_l[0])
      nodes[node['index']] = node
    booster[i] = nodes
  return booster

def get_leaf_index(nodes, x):
#   print(f"nodes is {nodes}")
  n = nodes[-1]
#   print(f"n is {n}")
#   print(f"x is {x}")
#   print(f"x.shape is {x.shape}")
#   print(f"x.shape[1] is {x.shape[1]}")
  while n['node_type'] == 'decision':
    print(f"n is {n}")
    col = n['split_col']
    val = n['split_val']
    if x[col] <= val:
      print("x[col] is less than split value, inside left node")
      print(f"x[col] is {x[col]}")
      print(f"val is {val}")
      next_node = n['left']
      if next_node == 0:
        next_node = n['right']
        if next_node == 0:
          next_node = 1
    else:
      print("x[col] is greater than split value, inside right node")
      print(f"x[col] is {x[col]}")
      print(f"val is {val}")
      next_node = n['right']
      if next_node == 0:
        next_node = n['left']
        if next_node == 0:
          next_node = 1
    n = nodes[next_node]
  print(f"returning {n['index']}")
  return n['index']

# retrieve the number of training samples that end up in each leaf of an estimator
def get_n_samples_in_leaf_of_estimator(num_trees,train_leaves):
    n_samples_in_leaf_of_estimator = []
    for t in range(num_trees):
        unique, counts = np.unique(train_leaves[:,t], return_counts=True)
        d = dict(zip(unique, counts))
        n_samples_in_leaf_of_estimator.append(d)
    return n_samples_in_leaf_of_estimator

# retrieve the weights of a new arriving observation
def get_model_weights(model, num_trees, n_samples_in_leaf_of_estimator, x_in,train_leaves):
    weights = dict()
    tree_weights = np.zeros(train_leaves.shape[0])
    if type(model).__name__ == 'GBDTMulti':
        leaf_id_pred = gbt_apply(x_in, num_trees).flatten()
    else:
        leaf_id_pred = model.apply(x_in).flatten()
    for i,leaf_id_x_i in enumerate(train_leaves):
        for t in range(num_trees):
            if  leaf_id_x_i[t] == leaf_id_pred[t]:
                tree_weights = 1/n_samples_in_leaf_of_estimator[t][leaf_id_pred[t]]    
                tree_weights /= num_trees   
                if i in weights.keys():
                    w = weights[i] + tree_weights
                else:
                    w = tree_weights
                weights[i] = w
    return weights


# get the number of estimators in the ensemble
def get_number_of_trees(model):
    if isinstance(model, XGBRegressor):
        dump_list = model.get_booster().get_dump()
        num_trees = len(dump_list)
    elif isinstance(model,(RandomForestRegressor,AdaBoostR2)):
        num_trees = model.n_estimators
    else:
        booster_list = tree_file_to_list('tree.txt')
        num_trees = len(booster_list)
    return num_trees

def gbt_apply(x_in, num_trees):
  booster_list = tree_file_to_list('tree.txt')
  with open(r'booster_list.txt', 'w') as fp:
    for booster in booster_list:
        fp.write("%s\n" % booster)
  booster_dict = booster_list_to_booster_dict(booster_list)
#   print(f'booster dict is:\n {booster_dict}')
  with open("booster_dict.json", "w") as outfile:
    json.dump(booster_dict, outfile, indent=2)         
  x_leaf = np.empty((x_in.shape[0],num_trees))
  for x_i,x in enumerate(x_in):
    for b_i in range(num_trees):
      print(f"current x_i is {x_i}")
      leaf = get_leaf_index(booster_dict[b_i],x)
      x_leaf[x_i,b_i] = leaf
  return x_leaf

def get_train_leaves(model, X_train, num_trees):
    if type(model).__name__ == 'GBDTMulti':
        train_leaves = gbt_apply(X_train, num_trees)
    else:
        train_leaves = model.apply(X_train)
    return train_leaves
    
# check if weights add up to 1
def check_sum(weights):
    if np.round(sum(weights.values()),2) != 1:
        raise ValueError('weights do not sum to 1')

# fill the weights of scenarios with a probability of 0
def fill_weights(weights,X_train):
    for i,x in enumerate(X_train):
        if i not in weights.keys():
            weights[i] = 0
    return weights  

# if necessary multiply by the learning rate used for training the model
def multiply_by_lr(weights,hp):
    if hp["learning_rate"] == "NaN":
        # print("worked")
        return weights
    else:
        weights.update({k: v * hp["learning_rate"] for k, v in weights.items()})
        return weights

# if the first estimator of the ensemble only contains a single leaf, add 1/D to the weights
# def account_for_single_leaf(weights, D):
#     weights.update({k: v + (1/D) for k, v in weights.items()})
#     return weights

# manipulate the weights dependend on the traning set size
def manipulate_weights(weights, Xi):
    weights.update({k:v**(Xi) for k,v in weights.items()})
    return weights

def normalize_weights(weights):
    weights.update({k:v/sum(weights.values()) for k,v in weights.items()}) 
    for k in weights.keys():
        if weights[k]<10**-6:
            weights[k]=5*10**-3
    weights.update({k:v/sum(weights.values()) for k,v in weights.items()}) 
    return weights

def weights_processing(weights,X_train,hp,D):
    weights = fill_weights(weights,X_train)
    weights = multiply_by_lr(weights,hp)
    # if isinstance(model, (AdaBoostR2, XGBRegressor,GradientBoostingRegressor)):
    #     weights = account_for_single_leaf(weights,D)
    xi = hp["Xi"]
    weights = normalize_weights(weights)
    weights = manipulate_weights(weights,xi)
    weights = normalize_weights(weights)
    check_sum(weights)
    return weights


def get_model_policy(model,X_train,X_test,hp):
    D = X_train.shape[0]
    num_trees = get_number_of_trees(model)
    train_leaves = get_train_leaves(model, X_train, num_trees)
    n_samples_in_leaf_of_estimator = get_n_samples_in_leaf_of_estimator(num_trees,train_leaves)
    weights = get_model_weights(model,num_trees,n_samples_in_leaf_of_estimator,X_test,train_leaves)
    weights = weights_processing(weights,X_train,hp,D)
    return weights


def train_random_forest_model_with_hp(hp,X_train,y_train):
    n_estimators = hp["n_estimators"]
    max_depth = hp["max_depth"]
    max_features = hp["max_features"]
    rf = RandomForestRegressor(n_estimators=n_estimators,
                                max_depth=max_depth,
                                max_features = max_features)
    rf.fit(X_train,y_train)
    return rf


def train_ada_boost_model_with_hp(hp,X_train,y_train):
    n_estimators = hp["n_estimators"]
    max_depth = hp["max_depth"]
    ada = AdaBoostR2(X_train,y_train,n_estimators=n_estimators,max_depth=max_depth)
    ada.fit()
    return ada


def train_gradient_boost_model_with_hp(hp,X_train,y_train):
    n_est = hp["n_estimators"]
    max_depth = hp["max_depth"]
    learning_rate = hp["learning_rate"]
    min_samples_split = hp["min_samples_split"]
    gbt_params = {
        "lr" : learning_rate,
        # "min_samples" : min_samples_split,
        "max_depth" : max_depth
    }
    gbdtmo_path = os.path.join(os.path.expanduser('~'), 'GBDTMO', 'build', 'gbdtmo.so')
    LIB = load_lib(gbdtmo_path)
    y_train_gbt = np.ascontiguousarray(y_train)
    X_train_gbt = np.ascontiguousarray(X_train)
    out_dim = y_train_gbt.shape[1]
    print(f"out dim is {out_dim}")
    grad =GBDTMulti(LIB, out_dim=out_dim, params=gbt_params)
    grad.set_data((X_train_gbt, y_train_gbt))
    print("Data is set")
    grad.train(n_est)
    print("Model is trained")
    grad.dump(b"tree.txt")
    graph = create_graph("tree.txt", 34, [0, 3])
    graph.render("tree_34", format='pdf')
    return grad


def train_xgboost_model_with_hp(hp,X_train,y_train):
    n_estimators = hp["n_estimators"]
    learning_rate = hp["learning_rate"]
    max_depth = hp["max_depth"]
    min_split_loss = hp["min_split_loss"]
    colsample_bynode = hp["max_features"]
    xgb = XGBRegressor(n_estimators=n_estimators, 
                                            learning_rate=learning_rate, 
                                            max_depth = max_depth,
                                            min_split_loss = min_split_loss,
                                            colsample_bynode = colsample_bynode)
    xgb.fit(X_train,y_train)
    return xgb

def train_model_with_hp(model,hyperparameters,X_train,y_train):
    
    if model == 'adaboost':
        model = train_ada_boost_model_with_hp(hyperparameters,X_train,y_train)
    
    elif model == 'rf':
        model = train_random_forest_model_with_hp(hyperparameters,X_train,y_train)

    elif model == 'gbt':
        model = train_gradient_boost_model_with_hp(hyperparameters,X_train,y_train)

    elif model == 'xgboost':
        model = train_xgboost_model_with_hp(hyperparameters,X_train,y_train)
    
    return model


def write_oos_loads_to_json(json_data, y_test,n,filename,y_scaler):
    buses = json_data["Buses"].keys()
    load = y_scaler.inverse_transform(y_test[n-1].reshape(1,-1))
    load = np.round(load,2).reshape(-1)
    b_i = 0
    for b in buses:
        if json_data["Buses"][b]["s1"]["Load (MW)"] != [0]*24 and json_data["Buses"][b]["s1"]["Load (MW)"] !=0: 
            bus_load = load[b_i * 24 : (b_i + 1) * 24]
            json_data["Buses"][b]["s1"]["Load (MW)"] = bus_load.tolist()
            b_i=b_i + 1
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_5min_oos_loads_to_json(json_data, y_test,n,filename,y_scaler):
    json_data["Parameters"]["Time step (min)"] = 5
    buses = json_data["Buses"].keys()
    load = y_scaler.inverse_transform(y_test[n-1].reshape(1,-1))
    load = np.round(load,2).reshape(-1)
    b_i = 0
    for b in buses:
        if json_data["Buses"][b]["s1"]["Load (MW)"] != [0]*288 and json_data["Buses"][b]["s1"]["Load (MW)"] != 0: 
            bus_load = load[b_i * 288: (b_i + 1) * 288]
            json_data["Buses"][b]["s1"]["Load (MW)"] = bus_load.tolist()
            b_i=b_i + 1
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_scenario_loads_to_json(json_data, y_train, pol, filename,y_scaler):
    buses = json_data["Buses"].keys()
    scenarios = json_data["Buses"]["b1"].keys()
    for s_i, s in enumerate(scenarios):
        load = y_scaler.inverse_transform(y_train[s_i].reshape(1,-1))
        load = np.round(load,2).reshape(-1)
        b_i = 0
        for b in buses:
            if json_data["Buses"][b][s]["Load (MW)"] != [0]*24 and json_data["Buses"][b][s]["Load (MW)"] != 0: 
                bus_load = load[b_i * 24 : (b_i + 1) * 24]
                json_data["Buses"][b][s]["Load (MW)"] = bus_load.tolist()
                b_i=b_i + 1
            json_data["Buses"][b][s]["Probability"] = float(np.round(pol[s_i],10))
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_naive_scenario_loads_to_json(json_data, y_train, filename,y_scaler,n_sns):
    buses = json_data["Buses"].keys()
    scenarios = json_data["Buses"]["b1"].keys()
    for s_i, s in enumerate(scenarios):
        # write scenario loads
        load = y_scaler.inverse_transform(y_train[s_i].reshape(1,-1))
        load = np.round(load,2).reshape(-1)
        b_i = 0
        for b in buses:
            if json_data["Buses"][b][s]["Load (MW)"] != [0]*24 and json_data["Buses"][b][s]["Load (MW)"] != 0: 
                bus_load = load[b_i * 24 : (b_i + 1) * 24]
                json_data["Buses"][b][s]["Load (MW)"] = bus_load.tolist()
                b_i=b_i + 1
            json_data["Buses"][b][s]["Probability"] = float(np.round(1/n_sns,10))
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def write_scenario_loads_to_json_for_point_prediction(json_data,y_hat,filename,y_scaler):
    buses = json_data["Buses"].keys()
    scenarios = json_data["Buses"]["b1"].keys()
    # print(f'y_hat dimension is {y_hat.shape}')
    for s_i, s in enumerate(scenarios):
        load = y_scaler.inverse_transform(y_hat[s_i].reshape(1,-1))
        load = np.round(load,2).reshape(-1)
        b_i = 0
        for b in buses:
            if json_data["Buses"][b][s]["Load (MW)"] != [0]*24 and json_data["Buses"][b][s]["Load (MW)"] != 0: 
                bus_load = load[b_i * 24 : (b_i + 1) * 24]
                json_data["Buses"][b][s]["Load (MW)"] = bus_load.tolist()
                b_i=b_i + 1
            json_data["Buses"][b][s]["Probability"] = 1.0
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def run_hp_experiment(m, hp, X_train, X_test, y_train,n_oos,n_scenarios,y_scaler, INPUT_FILE_DIR, Xi_list, X_train_dates):
    print(f"\n=== starting experiment with {m} model ===")
    print(f"--- number of scenarios {n_scenarios} ---")
    print(f'--- n_estimators {hp["n_estimators"]} ---')
    print(f'--- max_depth {hp["max_depth"]} ---')
    print(f'--- learning_rate {hp["learning_rate"]} ---')
    print(f'--- min split loss {hp["min_split_loss"]} ---')
    print(f'--- min samples split {hp["min_samples_split"]} ---')
    print(f'--- max features {hp["max_features"]} ---')
    start = time.time()
    model = train_model_with_hp(m, hp, X_train, y_train)
    for oos_scenario in range(1,n_oos+1):
        n_es = str(hp["n_estimators"])
        xi = str(hp["Xi"])
        m_d = str(hp["max_depth"])
        l_r = str(hp["learning_rate"])
        m_s_l = str(hp["min_split_loss"])
        m_s_s = str(hp["min_samples_split"])
        m_f = str(hp["max_features"])
        filename = INPUT_FILE_DIR + f'snum_{str(n_scenarios)}/oos_{str(oos_scenario)}/point/{str(m)}/n_est_{n_es}_Xi_xi_'\
            + f'max_depth_{m_d}_lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'
        with open(filename) as json_file:
            json_data = json.load(json_file)
        print('--- point oos_scenario: ', oos_scenario, ' ---')
        x_in = X_test[oos_scenario-1].reshape(1, -1)
        print(x_in)
        print(type(x_in))
        print(x_in.shape)
        y_hat = model.predict(np.ascontiguousarray(x_in)) 
        write_scenario_loads_to_json_for_point_prediction(json_data, y_hat, filename, y_scaler)
        for xi in Xi_list:
            hp["Xi"] = xi
            pol = get_model_policy(model,X_train,x_in,hp)
            filename = INPUT_FILE_DIR + f'snum_{str(n_scenarios)}/oos_{str(oos_scenario)}/weighted/{str(m)}/n_est_{n_es}_Xi_{xi}_'\
            + f'max_depth_{m_d}_lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'
            with open(filename) as json_file:
                json_data = json.load(json_file)
            print('--- weighted oos_scenario: ', oos_scenario, ' ---')
            write_scenario_loads_to_json(json_data,y_train,pol,filename,y_scaler)
            plot_scenario_weights(m, pol, hp, X_train_dates, INPUT_FILE_DIR, n_scenarios, oos_scenario)
    end = time.time()
    print(f"=== experiment finished in {np.round(end-start,2)} s===")
    return model

def write_simulation_input_files(params, ts_data, target_data, target_data_5min,\
    load_scal, INPUT_FILE_DIR):
    five_min = params["five_min"]
    start_date_train = params["start_date_train"]
    start_date_oos = params["start_date_oos"]
    for scenario_number in params["number_of_scenarios"]: 
        print(f"\n=== starting experiment with {scenario_number} scenarios ===")
        X_train, X_test, y_train, y_test, X_scaler, y_scaler, X_train_dates =\
            prepare_forecast_data(ts_data, target_data, number_of_scenarios=scenario_number,\
                load_scaling = load_scal, number_of_oos_scenarios=params["number_of_oos_scenarios"], \
                    start_date_train = start_date_train, start_date_oos=start_date_oos )
        if five_min == "True":
            _, _, y_train_5min, y_test_5min, _, y_scaler_5min, _ =\
                prepare_forecast_data(ts_data, target_data_5min, number_of_scenarios=scenario_number,\
                    load_scaling = load_scal, number_of_oos_scenarios=params["number_of_oos_scenarios"], \
                        start_date_train = start_date_train, start_date_oos=start_date_oos)
        for oos_scenario in range(1,params["number_of_oos_scenarios"]+1):
            filename = INPUT_FILE_DIR + f'snum_{str(scenario_number)}/oos_{str(oos_scenario)}/naive/naive/naive.json'
            with open(filename) as json_file:
                json_data = json.load(json_file)
            write_naive_scenario_loads_to_json(json_data, y_train, filename,y_scaler, scenario_number)
            filename = INPUT_FILE_DIR + f'snum_{str(scenario_number)}/oos_{str(oos_scenario)}/oos_{str(oos_scenario)}.json'
            with open(filename) as json_file:
                json_data = json.load(json_file)
            if five_min == "True":
                write_5min_oos_loads_to_json(json_data, y_test_5min, oos_scenario, filename, y_scaler_5min)
            else:
                write_oos_loads_to_json(json_data,y_test,oos_scenario,filename,y_scaler)
        for ml_algorithm in params["ml_algorithms"]:   
            for n_estimators in params[ml_algorithm]["n_estimators"]:
                for max_depth in params[ml_algorithm]["max_depth"]:
                    for learning_rate in params[ml_algorithm]["learning_rate"]:
                        for min_split_loss in params[ml_algorithm]["min_split_loss"]:
                            for min_samples_split in params[ml_algorithm]["min_samples_split"]:
                                for max_features in params[ml_algorithm]["max_features"]:
                                    hp = {'n_estimators' : n_estimators,
                                            'Xi' : 'xi',
                                            'max_depth' : max_depth,
                                            'learning_rate': learning_rate,
                                            'min_split_loss': min_split_loss,
                                            'min_samples_split': min_samples_split,
                                            'max_features': max_features}
                                    model = run_hp_experiment(ml_algorithm, hp, X_train, X_test,y_train,\
                                        params["number_of_oos_scenarios"], scenario_number,y_scaler, INPUT_FILE_DIR,\
                                            params[ml_algorithm]["Xi"], X_train_dates)

def create_new_scenario_files_point(REF_FILE, INPUT_FILE_DIR, scenario_number,\
    oos_scenario, ml_algorithm, n_sns, hp):
    POINT_FILE_PATH = f'{INPUT_FILE_DIR}snum_{scenario_number}/oos_{oos_scenario}/point/{ml_algorithm}/'
    Path(POINT_FILE_PATH).mkdir(parents=True, exist_ok=True)
    filename = REF_FILE
    with open(filename) as json_file:
        json_data = json.load(json_file)
    json_data["Parameters"]["Scenario number"] = n_sns
    buses = list(json_data["Buses"].keys())
    for b in buses:
        copy_data = json_data["Buses"][b]["s1"]
        for s in range(2, n_sns + 1):
            json_data["Buses"][b][f"s{s}"] = copy_data
    for n_estimators in hp["n_estimators"]:
        for max_depth in hp["max_depth"]:
            for learning_rate in hp["learning_rate"]:
                for min_split_loss in hp["min_split_loss"]:
                    for min_samples_split in hp["min_samples_split"]:
                        for max_features in hp["max_features"]:
                            n_es = str(n_estimators)
                            m_d = str(max_depth)
                            l_r = str(learning_rate)
                            m_s_l = str(min_split_loss)
                            m_s_s = str(min_samples_split)
                            m_f = str(max_features)
                            filename = POINT_FILE_PATH + f'n_est_{n_es}_Xi_xi_max_depth_{m_d}_lear_rate_{l_r}'\
                                + f'_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'
                            with open(filename, 'w', encoding='utf-8') as f:
                                    json.dump(json_data, f, ensure_ascii=False, indent=3)

def create_new_scenario_files(REF_FILE, INPUT_FILE_DIR, oos_scenario, ml_algorithm, scenario_number, hp):
    SCENARIO_FILE_PATH = f'{INPUT_FILE_DIR}snum_{scenario_number}/oos_{oos_scenario}/weighted/{ml_algorithm}/'
    Path(SCENARIO_FILE_PATH).mkdir(parents=True, exist_ok=True)
    filename = REF_FILE
    with open(filename) as json_file:
        json_data = json.load(json_file)
    json_data["Parameters"]["Scenario number"] = scenario_number
    buses = list(json_data["Buses"].keys())
    for b in buses:
        copy_data = json_data["Buses"][b]["s1"]
        for s in range(2, scenario_number + 1):
            json_data["Buses"][b][f"s{s}"] = copy_data
    for n_estimators in hp["n_estimators"]:
        for Xi in hp["Xi"]:
            for max_depth in hp["max_depth"]:
                for learning_rate in hp["learning_rate"]:
                    for min_split_loss in hp["min_split_loss"]:
                        for min_samples_split in hp["min_samples_split"]:
                            for max_features in hp["max_features"]:
                                n_es = str(n_estimators)
                                xi = str(Xi)
                                m_d = str(max_depth)
                                l_r = str(learning_rate)
                                m_s_l = str(min_split_loss)
                                m_s_s = str(min_samples_split)
                                m_f = str(max_features)
                                filename = SCENARIO_FILE_PATH + f'n_est_{n_es}_Xi_{xi}_max_depth_{m_d}_'\
                                     + f'lear_rate_{l_r}_min_sp_l_{m_s_l}_min_samples_split_{m_s_s}_max_features_{m_f}.json'
                                with open(filename, 'w', encoding='utf-8') as f:
                                        json.dump(json_data, f, ensure_ascii=False, indent=3)

def create_naive_files(REF_FILE, INPUT_FILE_DIR, scenario_number, oos_scenario):
    NAIVE_FILE_PATH = f'{INPUT_FILE_DIR}snum_{scenario_number}/oos_{oos_scenario}/naive/naive/'
    Path(NAIVE_FILE_PATH).mkdir(parents=True, exist_ok=True)
    filename = REF_FILE
    with open(filename) as json_file:
        json_data = json.load(json_file)
    json_data["Parameters"]["Scenario number"] = scenario_number
    buses = list(json_data["Buses"].keys())
    for b in buses:
        copy_data = json_data["Buses"][b]["s1"]
        for s in range(2, scenario_number + 1):
            json_data["Buses"][b][f"s{s}"] = copy_data
    filename = NAIVE_FILE_PATH + 'naive.json'
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)

def create_oos_files(REF_FILE, INPUT_FILE_DIR, scenario_number, oos_scenario):
    OOS_FILE_PATH = f'{INPUT_FILE_DIR}snum_{scenario_number}/oos_{oos_scenario}/'
    Path(OOS_FILE_PATH).mkdir(parents=True, exist_ok=True)
    with open(REF_FILE) as json_file:
        json_data = json.load(json_file)
    filename = OOS_FILE_PATH + f'oos_{oos_scenario}.json'
    with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=3)
###
def create_files(params, REF_FILE, REF_FILE_OOS, INPUT_FILE_DIR):
    for scenario_number in params["number_of_scenarios"]:
        for oos_scenario in range(1,params["number_of_oos_scenarios"]+1):
            create_oos_files(REF_FILE_OOS, INPUT_FILE_DIR, scenario_number, oos_scenario)
            create_naive_files(REF_FILE, INPUT_FILE_DIR, scenario_number, oos_scenario)
            for ml_algorithm in params["ml_algorithms"]:
                create_new_scenario_files(REF_FILE, INPUT_FILE_DIR, oos_scenario, ml_algorithm,\
                    scenario_number, params[ml_algorithm])
                create_new_scenario_files_point(REF_FILE, INPUT_FILE_DIR, scenario_number,\
                    oos_scenario, ml_algorithm, 1, params[ml_algorithm])
###
# WEST, N.Y.C., LONGIL, DUNWOD, HUD VL, CENTRL, MHK VL, MILLWD, NORTH, GENESE, CAPITL