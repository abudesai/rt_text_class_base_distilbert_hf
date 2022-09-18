import numpy as np, pandas as pd
import os
import sys

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.classifier as classifier


# get model configuration parameters 
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema): 
        self.model_path = model_path
        self.data_schema = data_schema
        self.preprocessor = None
        self.model = None
        self.id_field_name = self.data_schema["inputDatasets"]["textClassificationBaseMainInput"]["idField"]  
        
    
    def _get_preprocessor(self): 
        if self.preprocessor is None: 
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
            return self.preprocessor
        else: return self.preprocessor
    
    def _get_model(self):
        if self.model is None:   
            self.model = classifier.load_model(self.model_path)
            # self.model = Classifier(self.model_path)
            return self.model
        else: return self.model
        
    
    def _get_predictions(self, data):                  
        preprocessor = self._get_preprocessor()
        model = self._get_model()
    
        
        if preprocessor is None:  raise Exception("No preprocessor found. Did you train first?")
        if model is None:  raise Exception("No model found. Did you train first?")
                            
        # transform data - returns tuple of X (array of word indexes) and y (None in this case)
        proc_data = preprocessor.transform(data)   
        
        preds = model.predict( proc_data )
        
        return preds    
    
    
    def predict_proba(self, data):       
        preds = self._get_predictions(data)
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)       
        preds_df = pd.concat( [ data[[self.id_field_name]].copy(), pd.DataFrame(preds, columns = class_names)], axis=1 )
        return preds_df 
    
    
    
    def predict(self, data):               
        preds = self._get_predictions(data)   
        class_names = pipeline.get_class_names(self.preprocessor, model_cfg)  
        
        preds_df = data[[self.id_field_name]].copy()        
        preds_df['prediction'] = pd.DataFrame(preds, columns = class_names).idxmax(axis=1)             
        return preds_df
