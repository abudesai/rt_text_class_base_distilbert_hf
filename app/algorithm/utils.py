
from pydoc import doc
import numpy as np, pandas as pd, random
import sys, os
import glob
import json
import torch as T



def set_seeds(seed_value=42):
    if type(seed_value) == int or type(seed_value) == float:          
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        T.manual_seed(seed_value)
    else: 
        print(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def get_data(data_path):     
    all_files = os.listdir(data_path) 
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))    
    input_files = [ os.path.join(data_path, file) for file in csv_files ]
    if len(input_files) == 0: raise ValueError(f'There are no data files in {data_path}.')
    raw_data = [ pd.read_csv(file) for file in input_files ]
    data = pd.concat(raw_data)
    return data


def get_data_schema(data_schema_path): 
    try: 
        json_files = list(filter(lambda f: f.endswith('.json'), os.listdir(data_schema_path) )) 
        if len(json_files) > 1: raise Exception(f'Multiple json files found in {data_schema_path}. Expecting only one schema file.')
        full_fpath = os.path.join(data_schema_path, json_files[0])
        with open(full_fpath, 'r') as f:
            data_schema = json.load(f)
            return data_schema  
    except: 
        raise Exception(f"Error reading data_schema file at: {data_schema_path}")  
      

def get_json_file(file_path, file_type): 
    try:
        json_data = json.load(open(file_path)) 
        return json_data
    except: 
        raise Exception(f"Error reading {file_type} file at: {file_path}")   


def get_hyperparameters(hyper_param_path): 
    hyperparameters_path = os.path.join(hyper_param_path, 'hyperparameters.json')
    if not os.path.exists(hyperparameters_path): 
        return {}  # if not hp file given, then return empty dictionary for hyperparameters and let the default hyperparameters work 
    return get_json_file(hyperparameters_path, "hyperparameters")


def get_model_config():
    model_cfg_path = os.path.join(os.path.dirname(__file__), 'config', 'model_config.json')
    return get_json_file(model_cfg_path, "model config")


def get_hpt_specs():
    hpt_params_path = os.path.join(os.path.dirname(__file__), 'config', 'hpt_params.json')
    return get_json_file(hpt_params_path, "HPT config")
    

def save_json(file_path_and_name, data):
    """Save json to a path (directory + filename)"""
    with open(file_path_and_name, 'w') as f:
        json.dump( data,  f, 
                  default=lambda o: make_serializable(o), 
                  sort_keys=True, 
                  indent=4, 
                  separators=(',', ': ') 
                  )

def make_serializable(obj): 
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return json.JSONEncoder.default(None, obj)


def print_json(result):
    """Pretty-print a jsonable structure"""
    print(json.dumps(
        result,
        default=lambda o: make_serializable(o), 
        sort_keys=True,
        indent=4, separators=(',', ': ')
    ))
    

def save_dataframe(df, save_path, file_name): 
    df.to_csv(os.path.join(save_path, file_name), index=False)


def get_resampled_data(data, max_resample, target_col):    
    # if some minority class is observed only 1 time, and a majority class is observed 100 times
    # we dont over-sample the minority class 100 times. We have a limit of how many times
    # we sample. max_resample is that parameter - it represents max number of full population
    # samples of the minority class. For this example, if max_resample is 3, then, we will only
    # repeat the minority class 2 times over (plus original 1 time). 
    classes, class_count = np.unique(data[target_col], return_counts=True)
    max_obs_count = max(class_count)
    
    resampled_data = []
    for class_, count in zip(classes, class_count):
        if count == 0: continue 
        # find total num_samples to use for this class
        size = max_obs_count if max_obs_count / count < max_resample else count * max_resample
        # if observed class is 50 samples, and we need 125 samples for this class, 
        # then we take the original samples 2 times (equalling 100 samples), and then randomly draw
        # the other 25 samples from among the 50 samples
        
        full_samples = size // count        
        idx = data[target_col] == class_
        for _ in range(full_samples):
            resampled_data.append(data.loc[idx])
            
        # find the remaining samples to draw randomly
        remaining =  size - count * full_samples   
        
        resampled_data.append(data.loc[idx].sample(n=remaining))
        
    resampled_data = pd.concat(resampled_data, axis=0, ignore_index=True)    
    
    # shuffle the arrays
    resampled_data = resampled_data.sample(frac=1.0, replace=False)
    return resampled_data