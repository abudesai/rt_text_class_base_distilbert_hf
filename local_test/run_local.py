import os, shutil
import sys
from textwrap import indent
import time
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score



sys.path.insert(0, './../app')
import algorithm.utils as utils 
import algorithm.model_trainer as model_trainer
import algorithm.model_server as model_server
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier

# paths to the input/outputs from ml_vol (volume mounted to docker, but here we are mimicking it locally)
inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")

data_path = os.path.join(inputs_path, "data")
train_data_path = os.path.join(data_path, "training", "textClassificationBaseMainInput")
test_data_path = os.path.join(data_path, "testing", "textClassificationBaseMainInput")


model_path = "./ml_vol/model/"
model_access_path = os.path.join(model_path, "model.save")
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")
# trained_model_path = os.path.join(model_path, "trained_model")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_outputs")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

# local dir to place evaluation results
test_results_path = "test_results"
if not os.path.exists(test_results_path): os.mkdir(test_results_path)

# change this to whereever you placed your local testing datasets
local_datapath = "./../../datasets" 

'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os, or python-version related issues, so beware.
'''

model_name= classifier.MODEL_NAME


def create_ml_vol():    
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "training": {
                        "textClassificationBaseMainInput": None
                    },
                    "testing": {
                        "textClassificationBaseMainInput": None
                    }
                }
            },
            "model": {
                "model_config": None,
                "artifacts": None,
            }, 
            
            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,                
            }
        }
    }    
    def create_dir(curr_path, dir_dict): 
        for k in dir_dict: 
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None: 
                create_dir(dir_path, dir_dict[k])

    create_dir("", dir_tree)


def copy_example_files(dataset_name):     
    # data schema
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}_schema.json", os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    # train data    
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}_train.csv", os.path.join(train_data_path, f"{dataset_name}_train.csv"))    
    # test data     
    shutil.copyfile(f"{local_datapath}/{dataset_name}/{dataset_name}_test.csv", os.path.join(test_data_path, f"{dataset_name}_test.csv"))    
   

def train_and_save_algo():        
    # Read hyperparameters 
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)    
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # get trained preprocessor, model, training history 
    preprocessor, model = model_trainer.get_trained_model(train_data, data_schema, hyper_parameters, model_artifacts_path)            
    # Save the processing pipeline   
    pipeline.save_preprocessor(preprocessor, model_artifacts_path)
    # Save the model 
    classifier.save_model(model, model_artifacts_path)
    print("done with training")


def load_and_test_algo(): 
    # Read data
    test_data = utils.get_data(test_data_path)   
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)    
    # instantiate the trained model 
    predictor = model_server.ModelServer(model_artifacts_path, data_schema)
    # make predictions
    # predictions = predictor.predict_proba(test_data, data_schema)
    predictions = predictor.predict_proba(test_data)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    # score the results
    test_key = get_test_key()
    results = score(test_key, predictions, data_schema)  
    print("done with predictions")
    return results


def get_test_key():
    test_key = pd.read_csv(f"{local_datapath}/{dataset_name}/{dataset_name}_test_key.csv")
    return test_key


def score(test_key, predictions, data_schema): 
    # we need to get a couple of field names in the test_data file to do the scoring 
    # we get it using the schema file
    id_field = data_schema["inputDatasets"]["textClassificationBaseMainInput"]["idField"]
    target_field = data_schema["inputDatasets"]["textClassificationBaseMainInput"]["targetField"]
        
    predictions.columns = [ str(c) for c in predictions.columns ]  
    pred_class_names = [c for c in predictions.columns[1:]    ]  
    
    predictions["__pred_class"] = pd.DataFrame(predictions[pred_class_names], columns = pred_class_names).idxmax(axis=1)  
    predictions = predictions.merge(test_key[[id_field, target_field]], on=[id_field])
    pred_probabilities = predictions[pred_class_names].copy()
    
    Y = predictions[target_field].astype(str)
    Y_hat = predictions["__pred_class"].astype(str)    
   
    accu = accuracy_score(Y , Y_hat)  
    f1 = f1_score(Y , Y_hat, average='weighted', labels=np.unique(Y_hat))    
    precision = precision_score(Y , Y_hat, average='weighted', labels=np.unique(Y_hat))         
    recall = recall_score(Y , Y_hat, average='weighted', labels=np.unique(Y_hat))    
    
    # -------------------------------------
    # auc calculation     
    obs_class_names =  list(set(Y))    
    # find classes in test data that were not observed in training data (if any)
    missing_classes = [c for c in obs_class_names if c not in pred_class_names]
    class_names = pred_class_names + missing_classes  
    for c in missing_classes: 
        # implicit probabilities are zero - model couldnt have predicted these classes
        pred_probabilities[c] = 0.0   
        
    name_to_idx_dict = {n:i for i,n in enumerate(class_names)}
    mapped_classes_true = Y.map(name_to_idx_dict)    
    
    
    if len(class_names) == 2:
        count_tup = np.unique(mapped_classes_true, return_counts=True)
        minority_class_idx = count_tup[0][np.argmin(count_tup[1])]
        if minority_class_idx == 0:
            mapped_classes_true = 1-mapped_classes_true
        auc = roc_auc_score(mapped_classes_true, predictions[class_names].values[:, minority_class_idx])
    else:
        auc = roc_auc_score(mapped_classes_true, predictions[class_names], 
            labels=np.arange(len(class_names)), average='weighted', multi_class='ovo')    
    
    # -------------------------------------        
    scores = { 
               "accuracy": np.round(accu,4), 
               "f1_score": np.round(f1, 4), 
               "precision": np.round(precision, 4), 
               "recall": np.round(recall, 4), 
               "auc_score": np.round(auc, 4), 
               "perc_pred_missing": np.round( 100 * (1 - predictions.shape[0] / test_key.shape[0]), 4)
               }
    return scores


def save_test_outputs(results, run_hpt, dataset_name):    
    df = pd.DataFrame(results) if dataset_name is None else pd.DataFrame([results])        
    df = df[["model", "dataset_name", "run_hpt", "num_hpt_trials", 
             "accuracy", "f1_score", "precision", "recall", "auc_score", "perc_pred_missing",
             "elapsed_time_in_minutes"]]
    print(df)
    file_path_and_name = get_file_path_and_name(run_hpt, dataset_name)
    df.to_csv(file_path_and_name, index=False)
    

def get_file_path_and_name(run_hpt, dataset_name): 
    if dataset_name is None: 
        fname = f"_{model_name}_results_with_hpt.csv" if run_hpt else f"_{model_name}_results_no_hpt.csv"
    else: 
        fname = f"{model_name}_{dataset_name}_results_with_hpt.csv" if run_hpt else f"{model_name}_{dataset_name}_results_no_hpt.csv"
    full_path = os.path.join(test_results_path, fname)
    return full_path


def run_train_and_test(dataset_name, run_hpt, num_hpt_trials):
    start = time.time()    
    # create the directory which imitates the bind mount on container
    create_ml_vol()   
    # copy the required files for model training    
    copy_example_files(dataset_name)   
    # train the model and save          
    train_and_save_algo()        
    # load the trained model and get predictions on test data
    results = load_and_test_algo()        
    
    end = time.time()
    elapsed_time_in_minutes = np.round((end - start)/60.0, 2)
    
    results = { **results, 
               "model": model_name, 
               "dataset_name": dataset_name, 
               "run_hpt": run_hpt, 
               "num_hpt_trials": num_hpt_trials if run_hpt else None, 
               "elapsed_time_in_minutes": elapsed_time_in_minutes 
               }
    
    print(f"Done with dataset in {elapsed_time_in_minutes} minutes.")
    return results


if __name__ == "__main__": 
    
    
    datasets = ["clickbait", "drug_reviews", "ecommerce_categories", "fake_job_postings", "hate_speech", 
                "movie_reviews", "musical_instruments", "newsgroups", "spam_text", "tweet_emotions"]
    datasets = ["spam_text"]
    
    all_results = []
    run_hpt = False
    for dataset_name in datasets:        
        print("-"*60)
        print(f"Running dataset {dataset_name}")
        results = run_train_and_test(dataset_name, run_hpt, num_hpt_trials=None)
        save_test_outputs(results, run_hpt, dataset_name)            
        all_results.append(results)
        print("-"*60)
                
    save_test_outputs(all_results, run_hpt, dataset_name=None)