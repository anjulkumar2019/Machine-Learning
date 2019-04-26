"""
This module does following things:
    1. Find categorical and numerical variables
    2. Find datatype of each variable
    3. Find features having missing values
    4. Discarding features having greater than 10% missing values
    5. Finding significant variables
"""
# Initialize libraries - Begins
import pandas as pd
import numpy  as np
from   scipy  import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import ExtraTreesClassifier

class file_opening:
    def file_open(self, name):
        try:
            data=pd.read_csv(name)
        except:
            print("can't open file")
            return(-1)
        return(data)

class var_characteristic_discovery:
    def variable_cleaning(self, data,column_name, col_cat, col_num, col_miss):
        for item in column_name:
            if item in col_miss:
                if item in col_cat:
                    data[item]=data[item].fillna('No_Value')
                else:
                    data[item]=data[item].fillna(data[item].median())
        data_cat=data[[col for col in col_cat]]
        data_num=data[[col for col in col_num]]
        data_cat=pd.get_dummies(data_cat)
        data_temp=data_num.join(data_cat, how='left')
        print('hi')
        return(data_temp)        
       
    def var_characteristic(self, data):
        column_name     = data.columns
        column_type     = {}
        column_type_val = {}
        column_type_mis = {}
        column_cat      = {}
        column_numeric  = {}
        row, col        = data.shape
        threshold_to_del=0.1
        for item in column_name:
            if data[item].dtype == 'object':
                column_cat[item]      = data[item].unique()
            else:
                column_numeric[item]  = data[item].unique()
            column_type_mis[item] = data[item].isnull().sum()/float(row)                
        
        for item in column_name:
            if column_type_mis[item] > threshold_to_del:
                if item in column_cat:
                    try:
                        column_cat.pop(item)
                        print(item, 'deleted')
                    except:
                        print(item, 'already deleted')
                else:
                    try:
                        column_numeric.pop(item)
                        print(item, 'deleted')
                    except:
                        print(item, 'already deleted')
                try:
                    data.drop(item, axis=1, inplace=True)
                    print(item, 'deleted from dataframe')
                except:
                    print(item, 'already deleted from dataframe') 
                try:
                    column_type_mis.pop(item)
                    print(item, 'deleted from column_type_mis')
                except:
                    print(item, 'already deleted from column_type_mis')   
        data=self.variable_cleaning(data,column_name,column_cat,column_numeric,column_type_mis)
        return(data)
#            print(column_numeric)


def main():
    filename = 'data.csv'
    predictor='Fraud Flag'
    file_open = file_opening()
    var_char  = var_characteristic_discovery()
    data=file_open.file_open(filename)
    data=var_char.var_characteristic(data)
    print(data.isnull().sum())
    var_select=variable_selection()
    var_select.selectkbest(data,predictor)
    var_select.extratreec(data,predictor)

if __name__ == '__main__':
    main()

    
