# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 19:53:05 2020

@author: Lindsay Turner

Random Forest Model 
"""

##############################################################################
# IMPORT PACKAGES
##############################################################################

import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

##############################################################################
# FUNCTIONS
##############################################################################

def nre_fun(x, y):
    nre = (x - y) / (x + y)
    return nre


# SET SEED IN UNDERSAMPLE_DS
def undersample_ds(x, classCol, nsamples_class):
    for i in np.unique(x[classCol]):
        if (sum(x[classCol] == i) - nsamples_class != 0):            
            xMatch = x[(x[classCol]).str.match(i)]
            x = x.drop(xMatch.sample(n = len(xMatch) - nsamples_class).index)
    return x


# changes Classnames into integers representing each class
def string_to_int(y):
    unique_y = np.unique(y)
    y = y.to_numpy()
    for i in range(len(y)):
        for j in range(len(unique_y)):
            if(y[i] == unique_y[j]):
                y[i] = j
                
    y = y.astype('int')
    unique_y = unique_y.astype('int')
    return y, unique_y
   
     
##############################################################################
# IMPORT DATA & CREATE DATAFRAME
##############################################################################
    
dfAll = pd.read_csv(r'C:/Users/linds/NOAA/rf_training/data_raw/training_data_1M_sub.csv')

# SEED VARIABLE HERE!
nsamples_class = 500
training_bc = undersample_ds(dfAll, 'Classname', nsamples_class)
#training_bc$Classname <- as.factor(training_bc$Classname)


green_red = nre_fun(training_bc['green'], training_bc['red'])
blue_coastal = nre_fun(training_bc['blue'], training_bc['coastal'])
NIR2_yellow = nre_fun(training_bc['NIR2'], training_bc['yellow'])
NIR1_red = nre_fun(training_bc['NIR1'], training_bc['red'])
rededge_yellow = nre_fun(training_bc['rededge'], training_bc['yellow'])
red_NIR2 = nre_fun(training_bc['red'], training_bc['NIR2'])
rededge_NIR2 = nre_fun(training_bc['rededge'], training_bc['NIR2'])
rededge_NIR1 = nre_fun(training_bc['rededge'], training_bc['NIR1'])
green_NIR1 = nre_fun(training_bc['green'], training_bc['NIR1'])
green_NIR2 = nre_fun(training_bc['green'], training_bc['NIR2'])
rededge_green = nre_fun(training_bc['rededge'], training_bc['green'])
rededge_red = nre_fun(training_bc['rededge'], training_bc['red'])
yellow_NIR1 = nre_fun(training_bc['yellow'], training_bc['NIR1'])
NIR2_blue = nre_fun(training_bc['NIR2'], training_bc['blue'])
blue_red = nre_fun(training_bc['blue'], training_bc['red'])

indices_df = pd.concat([green_red, blue_coastal, NIR2_yellow, NIR1_red,
                        rededge_yellow, red_NIR2, rededge_NIR2,
                        rededge_NIR1, green_NIR1, green_NIR2, rededge_green,
                        rededge_red, yellow_NIR1, NIR2_blue, blue_red],
                       axis = 1)

feature_names = ['green red', 'blue coastal', 'NIR2 yellow', 'NIR1 red',
              'rededge yellow', 'red NIR2', 'rededge NIR2', 'rededge NIR1',
              'green NIR1', 'green NIR2', 'rededge green', 'rededge red',
              'yellow NIR1', 'NIR2 blue', 'blue red']
indices_df.columns = feature_names
indices_df = indices_df * 10000
indices_df['Classname'] = pd.Series(training_bc['Classname'],
                                    index = indices_df.index)

##############################################################################
# RANDOM FOREST MODEL
##############################################################################

# X data for rf
features = indices_df

# y data for rf. The y data needs to be as integers for sklearn. 
labels, labels_name = string_to_int(indices_df['Classname'])


X_train, X_test, y_train, y_test = train_test_split(features[feature_names],
                                                    labels,
                                                    train_size = 0.9,
                                                    random_state = 42,
                                                    stratify = labels)

rf = RandomForestClassifier(n_estimators = 200,
                            max_features = 5,
                            random_state = 8)

rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
#confusionmatrix = pd.DataFrame(confusion_matrix(y_test, predicted),
#                               columns=labels_name,
#                               index=labels_name)

