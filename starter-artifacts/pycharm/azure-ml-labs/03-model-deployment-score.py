import json
import os
import numpy as np
import pandas as pd
from sklearn import linear_model 
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from azureml.core import Run
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
import pickle
from sklearn.externals import joblib

def init():
    try:
        # One-time initialization of predictive model and scaler
        from azureml.core.model import Model
        
        global trainedModel   
        global scaler

        model_name = "usedcarsmodel" 
        model_path = Model.get_model_path(model_name)
        print('Looking for models in: ', model_path)

        trainedModel = pickle.load(open(os.path.join(model_path,'usedcarsmodel.pkl'), 'rb'))
        
        scaler = pickle.load(open(os.path.join(model_path,'scaler.pkl'),'rb'))

    except Exception as e:
        print('Exception during init: ', str(e))

def run(input_json):     
    try:
        inputs = json.loads(input_json)

        #Scale the input
        scaled_input = scaler.transform(inputs)
        
        #Get the scored result
        prediction = json.dumps(trainedModel.predict(scaled_input).tolist())

    except Exception as e:
        prediction = str(e)
    return prediction