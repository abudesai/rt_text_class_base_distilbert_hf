import pprint
import sys

def get_preprocess_params(data_schema): 
    # initiate the pp_params dict
    pp_params = {}   
            
    # set the id attribute
    pp_params["id_field"] = data_schema["inputDatasets"]["textClassificationBaseMainInput"]["idField"]   
            
    # set the document_field attribute
    pp_params["document_field"] = data_schema["inputDatasets"]["textClassificationBaseMainInput"]["documentField"]   
    
    # set the target attribute
    pp_params["target_field"] = data_schema["inputDatasets"]["textClassificationBaseMainInput"]["targetField"]   
    
    # pprint.pprint(pp_params)   ; sys.exit()  
    return pp_params