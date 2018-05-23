#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 09:27:09 2018
Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 2: predicting housing price
Fetch housing data and unzip
@author: x220
"""

import os, tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/blob/master/"
#https://github.com/ageron/handson-ml/blob/master/datasets/housing/housing.tgz
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fFetchHousingData(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    ## tarfile cannot open the dowloaded file 
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    return

#fFetchHousingData()