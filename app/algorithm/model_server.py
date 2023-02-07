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

    def _get_model(self):
        if self.model is None:
            self.model = classifier.load_model(self.model_path)
        return self.model
        
    
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

    def predict_to_json(self, data): 
        predictions_df = self.predict_proba(data)
        predictions_df.columns = [str(c) for c in predictions_df.columns]
        class_names = predictions_df.columns[1:]

        predictions_df["__label"] = pd.DataFrame(
            predictions_df[class_names], columns=class_names
        ).idxmax(axis=1)

        # convert to the json response specification
        id_field_name = self.id_field_name
        predictions_response = []
        for rec in predictions_df.to_dict(orient="records"):
            pred_obj = {}
            pred_obj[id_field_name] = rec[id_field_name]
            pred_obj["label"] = rec["__label"]
            pred_obj["probabilities"] = {
                str(k): np.round(v, 5)
                for k, v in rec.items()
                if k not in [id_field_name, "__label"]
            }
            predictions_response.append(pred_obj)
        return predictions_response
