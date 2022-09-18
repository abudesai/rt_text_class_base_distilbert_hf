

from sklearn.pipeline import Pipeline
import sys, os
import joblib


import algorithm.preprocessing.preprocessors as preprocessors


PREPROCESSOR_FNAME = "preprocessor.save"


def get_preprocess_pipeline(pp_params, model_cfg):  

    pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
    pipe_steps = []       
    
    pipe_steps.append(
        (
            pp_step_names["LABEL_ENCODER"],
            preprocessors.CustomLabelEncoder( 
                target_col=pp_params['target_field'],
                dummy_label=model_cfg['target_dummy_val']
                ),
        )
    )  
    
    pipe_steps.append(
        (
            pp_step_names["COLUMNS_RENAMER"],
            preprocessors.ColumnsRenamer( 
                columns_map = {
                    pp_params['document_field']: "text",
                    pp_params['target_field']: "label",
                }
                ),
        )
    )  
               
    main_pipeline = Pipeline( pipe_steps )    
    
    return main_pipeline
    

def get_class_names(pipeline, model_cfg):
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]    
    lbl_binarizer_lbl = pp_step_names['LABEL_ENCODER']
    lbl_binarizer = pipeline[lbl_binarizer_lbl]
    class_names = lbl_binarizer.classes_
    return class_names

 
def save_preprocessor(preprocess_pipe, file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    try: 
        joblib.dump(preprocess_pipe, file_path_and_name)   
    except: 
        raise Exception(f'''
            Error saving the preprocessor. 
            Does the file path exist {file_path}?''')  
    return    
    

def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    if not os.path.exists(file_path_and_name):
        raise Exception(f'''Error: No trained preprocessor found. 
        Expected to find model files in path: {file_path_and_name}''')
        
    try: 
        preprocess_pipe = joblib.load(file_path_and_name)     
    except: 
        raise Exception(f'''
            Error loading the preprocessor. 
            Do you have the right trained preprocessor at {file_path_and_name}?''')
    
    return preprocess_pipe 
    