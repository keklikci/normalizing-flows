import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class preprocess_data:
    def __init__(self, scaler, fillna_vals, dropna_vals, drop_vals):
        self.scaler = scaler 
        self.fillna_vals = fillna_vals
        self.dropna_vals = dropna_vals 
        self.drop_vals = drop_vals
        
    def dropna_features(self, data):
        data = data.dropna(subset = self.dropna_vals)
        return data 
    
    def impute(self, data):
        for feature in self.fillna_vals: 
            data[feature] = data[feature].fillna(value = np.mean(data[feature]))
        return data 
    
    def drop_features(self, data):
        data.drop(self.drop_vals, axis=1, inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
    
    def encode_categorical(self, data):
        data = pd.get_dummies(data)
        return data 
    
    def scale(self, data):
        columns = data.columns
        data = self.scaler.fit_transform(data)
        data = pd.DataFrame(data,columns=columns) 
        return data

