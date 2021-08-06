import numpy as np 
import pandas as pd
from pathlib import Path
import xlrd 

class load_data:
    def __init__(self, filename, directory):
        self.filename = filename
        self.directory = directory
        
    def create_directory(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    def read_data(self, directory, filename):
        data_dir = self.directory + self.filename 
        wb = xlrd.open_workbook(data_dir, encoding_override='iso-8859-1')
        return pd.read_excel(wb)