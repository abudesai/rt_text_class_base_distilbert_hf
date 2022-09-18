
# import nltk
import sys , os
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# stop_words_path = os.path.join(os.path.dirname(__file__), 'stop_words.txt')
# stopwords = set(w.rstrip() for w in open(stop_words_path))
# porter_stemmer=nltk.PorterStemmer()


 
class CustomLabelEncoder(BaseEstimator, TransformerMixin): 
    def __init__(self, target_col, dummy_label) -> None:
        super().__init__()
        self.target_col = target_col
        self.dummy_label = dummy_label
        self.lb = LabelEncoder()


    def fit(self, data):                
        self.lb.fit(data[self.target_col])             
        self.classes_ = self.lb.classes_ 
        return self 
    
    
    def transform(self, data):
        try: 
            check_val_if_pred = data[self.target_col].values[0]
        except:
            return data
        else:
            if self.target_col in data.columns and check_val_if_pred != self.dummy_label: 
                data[self.target_col] = self.lb.transform(data[self.target_col])
            else:
                return None      
            return data
        

class ColumnsRenamer(BaseEstimator, TransformerMixin): 
    def __init__(self, columns_map) -> None:
        super().__init__()
        self.columns_map = columns_map
        
    def fit(self, data): return self
    
    def transform(self, data): 
        data.rename(columns=self.columns_map, inplace=True)
        return data
    
