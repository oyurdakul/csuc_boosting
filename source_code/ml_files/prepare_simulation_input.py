from utils import *
import shutil
import json
from pathlib import Path
import os

with open("params.json") as json_file:
    params = json.load(json_file)
# paths
INPUT_FILE_DIR = 'input_files/'

ts_data = pd.read_csv(f'data/{params["system"]}/ts_data_feature_selection_red_rfe.csv', index_col = 0, parse_dates=['Datetime'])
target_data = pd.read_csv(f'data/{params["system"]}/target_data.csv', index_col = 0, parse_dates=['Datetime'])
target_data_5min = pd.read_csv(f'data/{params["system"]}/target_data_5min.csv', index_col = 0, parse_dates=['Datetime'])

if params["system"] == 'caiso':
    #max load in the caiso dataset: 41330 MW
    #peak load of the 14-bus system: 321.29 MW
    #total generation capacity of the 14-bus system: 765.31 MW
    load_scal = 765.31*0.80/41330
elif params["system"] == "nyiso":
    #max load in the nyiso dataset: 31866.4167 MW on Aug 29, 2018 at 5 pm
    load_scal = 765.31*0.33/31866.74
    # load_scal = 1

REF_FILE = REF_FILE_OOS = f'data/{params["system"]}.json'

if params["five_min"] == "True":
    REF_FILE_OOS = f'data/{params["system"]}_5min.json'

create_files(params, REF_FILE, REF_FILE_OOS, INPUT_FILE_DIR )
write_simulation_input_files(params, ts_data, target_data, target_data_5min,\
    load_scal, INPUT_FILE_DIR)
print("-----Simulation data prepared-----")
                