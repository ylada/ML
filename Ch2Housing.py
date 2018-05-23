#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:09:24 2018
Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 2: predicting housing price
@author: x220
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


HOUSING_PATH = "datasets/housing"
#HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


"""
------------            Load Data          -------------------------
# 1. load
# 2. observe
    2.1  df.head()  .hist()   .describe()   .info()
    2.2  df["income"].value_counts()  df.income.value_counts()
# 3.  modify
    3.1  add column to df:   df["new_col"]=np.ceil(df["income"]*6.3)
    3.2  drop column: new=df.drop("income", axis=1); doesnot change df itself
    3.3  drop rows:
    3.4  change values:
------------            End of Load Data   -------------------------
"""

def f_load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

   
housing = f_load_housing_data()
#housing.head()
#housing.hist(bins=50, figsize=(20,15))
#housing.describe()
#housing.info()
#housing["medican_income"].value_counts()
plt.show()    
# categorify medican income to 5
housing["income_cal"] = np.ceil(housing["median_income"]/1.5)
# cap to 5 for all above $75k
housing[housing["income_cal"]>5] = 5


"""
------------            Prepare Data          -------------------------
# 1. Splitting
    1.1  sklearn.model_selection.train_test_split(data, test_size, random_state)
         sklearn only handle numpy, not pandas dataframe
    1.2  sklearn.model_selection.StratiefiedShuffleSplit()
        create an instance, split data keeping its statistical distribution
  2. Visulization and investigation
    2.1  plot multi-dimension in 2d  
        df.plot(scatter/x/y/alpha/radius/color)
    2.2  correlation   
        df.corr() matrix of all numeric columns
        df.Column1.corr(df.Column2); df.loc[3,:].corr(df.loc[5,:])
    2.3  correlation:   pandas.tools.plotting.scatter_matrix()
          scatter_matrix(df[attributes]);  attributes=['price', 'volume', 'time']
  3. Cleaning
    3.1  define train, train label, test, test label
    3.2  missing features
        3.2.1 remove the row/column, or fill with value (zero, mean, median, etc)
        df.drop("total_bedrooms", axis=1)    #drop column(axis1) or row (axis0)
        df["income"].fillna(df["income"].median())  #fill na with median
           #! need to save the mdeidan value to fill test, and new data
        3.2.2 from sklearn.prepocessing import Imputer
        imputer = Imputer(strategy="median")
        df_num = df.drop("cols that not numeric", axis=1) #imputer need numberic
        imputer.fit(df_num)
        print(imputer.statistics_)
        x=imputer.transform(df_num)  #transform and turn into numpy array
    3.3 encode text and categorical attributes
        3.3.1 # distance of simple encode may affect: 2,3 more related than 5
          from sklearn.preprocessing import LabelEncoder
          encoder = LabelEncoder
          df_cat = df["ocean"].apply(str)  #covert series to np string
          df_encoded=encoder.fit_transoform(df_cat)
          print(encoder.classes_)
        3.3.2 # OneHotEncoder avoids relevant numbers using binary column/attrib
              # Do above encoding first, then do OneHot
          from sklearn.preprocessing import OneHotEncoder
          encoder = OneHotEncoder()
          df_1hot = encoder.fit_transform(df_encoded.reshape(-1,1)) #sparse matr
          df_1hot.toarray()
        3.3.3 #combine the above two
          from sklearn.preprocessing import LabelBinarizer
          encoder = LabelBinarizer()    #return NP array unless sparse_out=True 
          df_1hot=encoder.fit_transform(df["cat"]) #cat is the column with text
          type(df_1hot)    #is numpy array
    3.4 Feature Scaling
        3.4.1  min-max scaling using MinMaxScaler() or (x-min)/(max-min)
        3.4.2  standard scaling using StandardScaler()
        
------------            End of Prepare Data   -------------------------
"""

#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Stratfied split of pandas dataframe to keep statistics of the median_income
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cal"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cal"], axis=1, inplace=True)

#check stats of the income
print(housing["income_cal"].value_counts() / len(housing)) 

housing = strat_train_set.copy()
housing = housing[housing["longitude"] < 0]
#housing.drop(housing.longitude > 0)
#housing.plot(kind="scatter", x='longitude', y='latitude', alpha=0.1)
#show multidimension info using x, y, size, color, alpha,
housing.plot(kind="scatter", x="median_income", y="median_house_value", 
             alpha=0.4,
             s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            )
plt.legend()
#correlation of the columns
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#from pandas.tools.plotting import scatter_matrix
#attributes = ["median_house_value", "median_income", "total_rooms", 
#              "housing_median_age"]
#scatter_matrix(housing[attributes], figsize=(12,8))

#define training features and labels
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#Clean null/missing values
#housing["total_bedrooms"].fillna(housing["total_bedrooms"].median())
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)   #must be all numeric
imputer.fit(housing_num)    #fit training data, to process train and test data
print(imputer.statistics_)  #calculate all median for all numberic columns
X = imputer.transform(housing_num) #transform training data with missing value

# convert text labels to numbers, simple encode; complication with numbers
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#housing_cat = housing["ocean_proximity"].apply(str) #encode string, not series
#housing_cat_encoded = encoder.fit_transform(housing_cat) 

# OneHotEncode, turn binary attributes/columns, number value doesnot matter
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()    #return NP array unless sparse_out=True 
housing_cat = housing["ocean_proximity"].apply(str) #encode string, not series
housing_1hot=encoder.fit_transform(housing_cat) 
