#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import pprint

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
from algorithm.model.classifier import Classifier, get_data_based_model_params
from algorithm.utils import get_model_config



# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params, save_path):  
    
    # set random seeds
    utils.set_seeds()
    
    # balance the target classes  
    data = utils.get_resampled_data(data = data, 
                max_resample = model_cfg["max_resample_of_minority_classes"],
                target_col=data_schema["inputDatasets"]["textClassificationBaseMainInput"]["targetField"]  
                )

    # perform train/valid split 
    train_data, valid_data = train_test_split(data, test_size=model_cfg['valid_split'])
    # print(train_data.shape, valid_data.shape) #; sys.exit()    

    # preprocess data
    print("Pre-processing data...")
    train_data, valid_data, preprocessor = preprocess_data(train_data, valid_data, data_schema)
    # print("train/valid data shape: ", train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)      
    
    # Create and train model   
    model = train_model(train_data, valid_data, hyper_params, save_path)    
    return preprocessor, model


def train_model(train_data, valid_data, hyper_params, save_path):    
    # get model hyper-parameters parameters 
    data_based_params = get_data_based_model_params(train_data, valid_data, hyper_params)
    
    model_params = { **data_based_params, **hyper_params}
    # pprint.pprint(model_params)

    # Create and train model   
    model = Classifier(  **model_params )  
      
    print('Fitting model ...')  
    trained_model = model.fit(
        train_data, valid_data, 
        save_path=save_path,
        num_train_epochs=2, 
    )
          
    return trained_model


def preprocess_data(train_data, valid_data, data_schema):    
    
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(data_schema)  
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)    
    
    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)  

    return train_data, valid_data, preprocess_pipe

