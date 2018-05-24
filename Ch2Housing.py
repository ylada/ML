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
  1. Splitting
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
    3.4 Other transforms (add/drop/merge attributes using transformers)
        a class with fit() transform() and fit_transform() help automation
        from sklearn.base importBaseEstimator, TransformerMixin
        class MyAddAttributes(BaseEstimator, TransformerMixin)
            def __init__(self, p1):
                self.p1 = p1
            def fit(self, X, y=None): 
                return self  # do nothing
            def transform(self, X, y=None)
                cost_per_unit=X[:, 3]/p1 #assume col3 all cost, p1 unit number
                return np.c_[X, cost_per_unit] # add column, in numpy
        attr_add = MyAddAttribute(p1=100)
        data = attr_add.transform(df.values) #add column, dataframe to np array
    3.5 Feature Scaling
        3.5.1  min-max scaling using MinMaxScaler() or (x-min)/(max-min)
        3.5.2  standard scaling using StandardScaler()
    3.6 Pipeline all pre-processing data
        3.5.1 fit and transform for all except last estimator  only fit()
------------            End of Prepare Data   -------------------------
"""

# 1. Data Split
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Stratfied split of pandas dataframe to keep statistics of the median_income
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cal"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cal"], axis=1, inplace=True)

# 2.1 Visualize
#check stats of the income
print(housing["income_cal"].value_counts() / len(housing)) 

housing = strat_train_set.copy()
housing = housing[housing["longitude"] < 0]
#housing.drop(housing.longitude > 0)
#housing.plot(kind="scatter", x='longitude', y='latitude', alpha=0.1)
#show multidimension info using x, y, size, color, alpha,
#housing.plot(kind="scatter", x="median_income", y="median_house_value", 
#             alpha=0.4,
#             s=housing["population"]/100, label="population",
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#            )
#plt.legend()

# 2.2 correlation of attributes
#corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))
#from pandas.tools.plotting import scatter_matrix
#attributes = ["median_house_value", "median_income", "total_rooms", 
#              "housing_median_age"]
#scatter_matrix(housing[attributes], figsize=(12,8))

# 3.1 Cleaning - define training features and labels
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 3.2 Cleaning - null/missing values
#housing["total_bedrooms"].fillna(housing["total_bedrooms"].median())
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)   #must be all numeric
imputer.fit(housing_num)    #fit training data, to process train and test data
#print(imputer.statistics_)  #calculate all median for all numberic columns
#imputer.transform() returns numpy array
X = imputer.transform(housing_num) #transform training data with missing value

# 3.3.1 Cleaning - convert text attributes to numbers
#                simple encode; complication with numbers
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#housing_cat = housing["ocean_proximity"].apply(str) #encode string, not series
#housing_cat_encoded = encoder.fit_transform(housing_cat) 
# 3.3.2 Clearning - one hot encode to convert text attributes
# OneHotEncode, turn binary attributes/columns, number value doesnot matter

#from sklearn.preprocessing import OneHotEncoder
#from future_encoders import OneHotEncoder
#for now import from future_encoders.py
#encoder2 = OneHotEncoder()    #need to do LabelEncoder first
#housing_cat_1hot = encoder2.fit_transform(housing_cat_encoded.reshape(-1,1))
# the follwoing combines abover text to numeric then to one hot encode
# Note: LabelBinarizer only converts integer, will be replaced by others
#from sklearn.preprocessing import LabelBinarizer
#encoder = LabelBinarizer()    #return NP array unless sparse_out=True 
#housing_cat = housing["ocean_proximity"].apply(str) #encode string, not series
#housing_1hot=encoder.fit_transform(housing_cat) 

# 3.4 add other transformers, good for automation pre-processing
#define _init_, fit, transform, fit_transform()
#BaseEstimator has , TransformorMinxin has auto fit_transform()
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6 #column numbers
class AddAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedroom_per_room = True):
        self.add_bedroom_per_room = add_bedroom_per_room
    def fit(self, X, y=None):
        return self    #do nothing
    def transform(self, X, y=None):
        rooms_per_household=X[:, rooms_ix] / X[:, household_ix]
        population_per_household=X[:, population_ix] / X[:, household_ix]
        if self.add_bedroom_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            #return numpy array
            return np.c_[X, rooms_per_household, population_per_household]
# create an instance for the class to add attributes/columns
attr_adder = AddAttributes(add_bedroom_per_room = False)
#needs np array in above transform(), not dataframe, return nump array
housing_extra_attribs = attr_adder.transform(housing.values) 


# 3.5  feature scaling (included in pipeline in #4)
from sklearn.preprocessing import StandardScaler
#standard_scaler = StandardScaler()
#housing_scaled = standard_scaler(housing.values)

# 3.6 Pipeline all pre-processing
#    FeaureUnion join different transforms/piplines together
from sklearn.pipeline import Pipeline, FeatureUnion
# build pipelines for numeric and text attributes, then join together
# following gets lists for numeric / text columns
num_attribs, cat_attribs = list(housing_num), ["ocean_proximity"]
# build a transformer to select attributes/columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attributes_names].values
# pipeline to transform numeric data
num_pipeline = Pipeline([
                         ('selector', DataFrameSelector(num_attribs)), #num col
                         ('imputer', Imputer(strategy="median")), #missing value
                         ('attribs_adder', AddAttributes()), #customize add col
                         ('std_scaler', StandardScaler()),  #feature scaling
                         ])
#the first transformer Imputer: input is dataframe, output is numpy array
#housing_num_tr = num_pipeline.fit_transform(housing_num)
#pipeline to transform text data
from future_encoders import OneHotEncoder, OrdinalEncoder
cat_pipeline = Pipeline([
                         ('selector', DataFrameSelector(cat_attribs)), #text col
                         ('cat_pipline', OneHotEncoder(sparse=False)), #turn text to num
                         ])
#join the features by two pipeline
full_pipeline = FeatureUnion(transformer_list=[
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
            ])
#housing_prepared = cat_pipeline.fit_transform(housing)
housing_prepared = num_pipeline.fit_transform(housing)
#encoder=OrdinalEncoder() #OneHotEncoder(sparse=False)
#b = housing[cat_attribs]
#a = encoder.fit_transform(b)
"""
------------            Train and Evaluate          -------------------------
  4. Training
    4.1  LinearRegression():  input: numpy arrary or dataframe, output: np array
        model = LinearRegression
        model.fit(train_X, train_y)
    4.2 Evaluate training performance: mean_squared_error in sklearn.metrics
        predict = model.predict(train_X)  #check training set, returns np array
        model_mse = mean_squared_error(predict, train_y)
        model_mse = np.sqrt(model_mse)
    4.3 Other models: Decision Tree(): RandomForest
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        model = DecisionTreeREgressor()  # or RandomFrestRegressor()
    4.4 Cross Validation: training set split n parts, train n-1, test 1
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, train_X, train_y   #10 parts
                                 scoreing='neg_mean_squared_error', cv=10)
        model_rmse_scores = np.sqrt(-scores) #return np array of 10
    4.5 Grid Search of hyperparameters, using cross validation(cv)
        !resist temptation to tweak hyperparameters to look good on test set!
        from sklearn.model_selection import GridSearchCV
        para_grid = [     # list of dicts, 
                     {'p1':[1,2,3], 'p2': [4,5]}, #1st try 3x2 combination
                     {'p1':[3,4], 'p3': [False]}, #2nd try 2 x 1 combination
                     ]                            # total 6+3 combination
        # construct the estimator
        grid_search = GridSerachCV(model, para_grid, cv=5,   
                                   scoring='neg_mean_squared_error')
        grid_search.fit(train_X, train_y)   #train the model
        grid_search.best_params_        # best parameter
        # cross validation evaluation scores, np array, are avaiable
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        final_model = grid_search.best_estimator_   # best estimator/parameter
        
    4.6 RandomizedSearchCV randomly search all hyperparameters for interations
    4.7 Analyze best models and importance of different attributes
        feature_importance = grid_search.best_estimator_.feature_importance_
        #list the importance with attributes' names
        sorted(zip(feature_importance, attributes), reverse=True) 
    4.8 Evaluate test set
        4.8.1 get the final best model
        final_model = grid_search.best_estimator_   # best estimator/parameter
        4.8.2 run pipeline to transform() test data, !NOT fit_transform()!
        test_X_prepared = full_pipeline.transform(test_X)
        4.8.3 predict
        final_prediction = final_model.predict(test_X_prepared)
        4.8.4 evaluation error
        final_mse = mean_square_error(test_y, final_prediction)
        final_rmse = np.sqrt(final_mse)
    4.7 save and load trained model using picle module
        from sklearn.externals import joblib
        joblib.dump(lin_reg, 'ch2_LinearModel.pkl')
        #load_Linear = joblib.load('ch2_LinearModel.pkl') #later load the model
        #load_Linear.predict(some_prepared_dataX)
        
------------            End of Train and Evaluate   -------------------------
"""
# 4.1 Training with Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
#try the model using part of training set
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
# all data feed to model need the same pre-processing
some_data_prepared = num_pipeline.fit_transform(some_data)
print('Predictions: \t', lin_reg.predict(some_data_prepared))
print('Labels: \t', list(some_labels))

# 4.2 Evaluate using mean_squared_error
from sklearn.metrics import mean_squared_error
#lin_predictions = lin_reg.predict(housing_prepared) #training set performance
#lin_mse = mean_squared_error(housing_labels, lin_predictions)
#lin_mse = np.sqrt(lin_mse)
#print('Linear Mean Square Error: \t', lin_mse)

# 4.3.1 Decision Tree
from sklearn.tree import DecisionTreeRegressor
#tree_reg = DecisionTreeRegressor()
#tree_reg.fit(housing_prepared, housing_labels)
#tree_predictions = tree_reg.predict(housing_prepared)
#tree_mse = mean_squared_error(housing_labels, tree_predictions)
#tree_mse = np.sqrt(tree_mse)
#print('Tree Mean Square Error: \t', tree_mse)
# 4.3.2 Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
#forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                 scoring='neg_mean_squared_error', cv=10)
#forest_rmse_score = np.sqrt(-forest_scores)
#display_scores(forest_rmse_score)

# 4.4 Cross Validation
#tree_reg is the model instance
from sklearn.model_selection import cross_val_score
#tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, #10 parts
#                         scoring='neg_mean_squared_error', cv=10)
#tree_rmse_scores = np.sqrt(-tree_scores)
def display_scores(scores):
    print("Scores:\t", scores)
    print("Mean: \t", scores.mean())
    print("Standard deviation: \t", scores.std())
display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# 4.5 Grid Search of hyperparameters
from sklearn.model_selection import GridSearchCV
para_grid = [
             {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
             {'bootstrap': [False], 'n_estimators': [3, 10], 
              'max_features': [2, 3, 4]},
             ]
#grid_search = GridSearchCV(forest_reg, para_grid, cv=5,
#                           scoring='neg_mean_squared_error')
#grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
grid_search.predict(housing_prepared)

# 4.7  Analyze best models and their errors/attibutes
feature_importance = grid_search.best_estimator_.feature_importances_
# list the importance of attributes with their names, decide which can go away
sorted(zip(feature_importance, num_attribs), reverse=True)

# 4.8 Evalute test set, through all preprocess as training, use transform()
final_model = grid_search.best_estimator_
test_X = strat_test_set.drop("median_house_value", axis=1)
test_y = strat_test_set["median_house_value"].copy()
test_X_prepared = num_pipeline.transform(test_X)
final_prediction = final_model.predict(test_X_prepared)
final_mse = mean_squared_error(test_y, final_prediction)
final_rmse = np.sqrt(final_mse)
print("final rms is: \t", final_rmse)

# 4.9 save and load trained model using picle module
from sklearn.externals import joblib
joblib.dump(final_model, 'ch2_finalmodel.pkl')
#load_forest = joblib