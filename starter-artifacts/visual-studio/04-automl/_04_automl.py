# Step 1 - Load training data and prepare Workspace
###################################################
import os
import numpy as np
import pandas as pd
from sklearn import linear_model 
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import azureml
from azureml.core import Run
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
import pickle

# Verify AML SDK Installed
# view version history at https://pypi.org/project/azureml-sdk/#history 
print("SDK Version:", azureml.core.VERSION)


# Load our training data set
print("Current working directory is ", os.path.abspath(os.path.curdir))
df_affordability = pd.read_csv('data/UsedCars_Affordability.csv', delimiter=',')
print(df_affordability)

full_X = df_affordability[["Age", "KM"]]
full_Y = df_affordability[["Affordable"]]

# Create a Workspace
#Provide the Subscription ID of your existing Azure subscription
subscription_id = "e223f1b3-d19b-4cfa-98e9-bc9be62717bc"

#Provide values for the Resource Group and Workspace that will be created
resource_group = "aml-workspace-z"
workspace_name = "aml-workspace-z"
workspace_region = 'westcentralus' # eastus, westcentralus, southeastasia, australiaeast, westeurope

# By using the exist_ok param, if the worskpace already exists we get a reference to the existing workspace instead of an error
ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)

print("Workspace Provisioning complete.")



# Step 2 - Define a helper method that will use AutoML to train multiple models and pick the best one
##################################################################################################### 
def auto_train_model(ws, experiment_name, model_name, full_X, full_Y,training_set_percentage, training_target_accuracy):

    # start a training run by defining an experiment
    experiment = Experiment(ws, experiment_name)
    
    train_X, test_X, train_Y, test_Y = train_test_split(full_X, full_Y, train_size=training_set_percentage, random_state=42)

    train_Y_array = train_Y.values.flatten()

    # Configure the automated ML job
    # The model training is configured to run on the local machine
    # The values for all settings are documented at https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train
    # Notice we no longer have to scale the input values, as Auto ML will try various data scaling approaches automatically
    Automl_config = AutoMLConfig(task = 'classification',
                                 primary_metric = 'accuracy',
                                 max_time_sec = 12000,
                                 iterations = 20,
                                 n_cross_validations = 3,
                                 exit_score = training_target_accuracy,
                                 blacklist_algos = ['kNN','LinearSVM'],
                                 X = train_X,
                                 y = train_Y_array,
                                 path='.\\outputs')

    # Execute the job
    run = experiment.submit(Automl_config, show_output=True)

    # Get the run with the highest accuracy value.
    best_run, best_model = run.get_output()

    return (best_model, run, best_run)


# Step 3 - Execute the AutoML driven training
#############################################
experiment_name = "Experiment-AutoML-04"
model_name = "usedcarsmodel"
training_set_percentage = 0.50
training_target_accuracy = 0.93
best_model, run, best_run = auto_train_model(ws, experiment_name, model_name, full_X, full_Y, training_set_percentage, training_target_accuracy)

# Examine some of the metrics for the best performing run
import pprint
pprint.pprint({k: v for k, v in best_run.get_metrics().items() if isinstance(v, float)})

# Step 4 - Try the best model
#############################
age = 60
km = 40000
print(best_model.predict( [[age,km]] ))

# Step 5 - Register the best performing model for later use and deployment
#################################################################
# notice the use of the root run (not best_run) to register the best model
run.register_model(description='AutoML trained used cars classifier')