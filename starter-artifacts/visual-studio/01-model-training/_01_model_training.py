# Step 1 - load the data
########################
import numpy as np
import pandas as pd
import os

print("Current working directory is ", os.path.abspath(os.path.curdir))
df = pd.read_csv('./data/UsedCars_Clean.csv', delimiter=',')
print(df)

# Step 2 - add the affordable feature
######################################
df['Affordable'] = np.where(df['Price']<12000, 1, 0)
df_affordability = df[["Age","KM", "Affordable"]]
print(df_affordability)

# Step 3 - Scale the numeric feature values
###########################################
X = df_affordability[["Age", "KM"]].values
y = df_affordability[["Affordable"]].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(pd.DataFrame(X).describe().round(2))
print(pd.DataFrame(X_scaled).describe().round(2))

# Step 4 - Fit a Logistic Regression
####################################
from sklearn import linear_model
# Create a linear model for Logistic Regression
clf = linear_model.LogisticRegression(C=1)

# we create an instance of Classifier and fit the data.
clf.fit(X_scaled, y)


# Step 5 - Test the trained model's prediction
##############################################
age = 60
km = 40000

scaled_input = scaler.transform([[age, km]])
prediction = clf.predict(scaled_input)

print("Can I afford a car that is {} month(s) old with {} KM's on it?".format(age,km))
print("Yes (1)" if prediction[0] == 1 else "No (0)")

# Step 6 - Measure the model's performance
##########################################
scaled_inputs = scaler.transform(X)
predictions = clf.predict(scaled_inputs)
print(predictions)

from sklearn.metrics import accuracy_score
score = accuracy_score(y, predictions)
print("Model Accuracy: {}".format(score.round(3)))


# Step 7 - Define a method to experiment with different training subset sizes
#############################################################################
from sklearn.model_selection import train_test_split
full_X = df_affordability[["Age", "KM"]]
full_Y = df_affordability[["Affordable"]]

def train_eval_model(full_X, full_Y,training_set_percentage):
    train_X, test_X, train_Y, test_Y = train_test_split(full_X, full_Y, train_size=training_set_percentage, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    clf = linear_model.LogisticRegression(C=1)
    clf.fit(X_scaled, train_Y)

    scaled_inputs = scaler.transform(test_X)
    predictions = clf.predict(scaled_inputs)
    score = accuracy_score(test_Y, predictions)

    return (clf, score)

# Step 8 - Verify AML SDK Installed
#####################################################################
import azureml.core
print("SDK Version:", azureml.core.VERSION)



# Step 9 - Create a workspace
#####################################################################

#Provide the Subscription ID of your existing Azure subscription
subscription_id = "e223f1b3-d19b-4cfa-98e9-bc9be62717bc"

#Provide values for the Resource Group and Workspace that will be created
resource_group = "aml-workspace-z"
workspace_name = "aml-workspace-z"
workspace_region = 'westcentralus' # eastus, westcentralus, southeastasia, australiaeast, westeurope


import azureml.core

# import the Workspace class 
from azureml.core import Workspace

ws = Workspace.create(
    name = workspace_name,
    subscription_id = subscription_id,
    resource_group = resource_group, 
    location = workspace_region,
    exist_ok = True)

print("Workspace Provisioning complete.")


# Step 10 - Create an experiment and log metrics for multiple training runs
###########################################################################
from azureml.core.run import Run
from azureml.core.experiment import Experiment

# start a training run by defining an experiment
myexperiment = Experiment(ws, "UsedCars_Experiment")
root_run = myexperiment.start_logging()

training_set_percentage = 0.25
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

training_set_percentage = 0.5
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

training_set_percentage = 0.75
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

training_set_percentage = 0.9
run = root_run.child_run("Training_Set_Percentage-%0.5F" % training_set_percentage)
model, score = train_eval_model(full_X, full_Y, training_set_percentage)
print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))
run.log("Training_Set_Percentage", training_set_percentage)
run.log("Accuracy", score)
run.complete()

# Close out the experiment
root_run.complete()

# Step 11 - Review captured runs
################################
# Go to the Azure Portal, find your Azure Machine Learning Workspace, select Experiments and select the UsedCars_Experiment

# You can also query the run history using the SDK.
# The following command lists all of the runs for the experiment
runs = [r for r in root_run.get_children()]
print(runs)

# Step 12 - Create an Azure Batch AI cluster
#############################################################################################
ws = Workspace.get(name=workspace_name, subscription_id=subscription_id,resource_group=resource_group)
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

experiment_name = "UsedCars_ManagedCompute_01"

from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster
batchai_cluster_name = "UsedCars-02"
cluster_min_nodes = 1
cluster_max_nodes = 3
vm_size = "STANDARD_DS11_V2"


if batchai_cluster_name in ws.compute_targets:
    compute_target = ws.compute_targets[batchai_cluster_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found existing compute target, using this compute target instead of creating:  ' + batchai_cluster_name)
    else:
        print("Error: A compute target with name ",batchai_cluster_name," was found, but it is not of type AmlCompute.")
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size, # NC6 is GPU-enabled
                                                                min_nodes = cluster_min_nodes, 
                                                                max_nodes = cluster_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, batchai_cluster_name, provisioning_config)
    
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
     # For a more detailed view of current BatchAI cluster status, use the 'status' property    
    print(compute_target.status.serialize())



# Step 13 - Upload the dataset to the DataStore
###############################################
ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)
ds.upload(src_dir='./data', target_path='used_cars', overwrite=True, show_progress=True)


# Prepare batch training script
# - See ./training/train.py
################################


# Step 14 - Create estimator
#############################
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
    '--training-set-percentage': 0.3
}

est_config = Estimator(source_directory='training',
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=['scikit-learn','pandas'])

# Step 15 - Execute the estimator job
#####################################
run = exp.submit(config=est_config)
run

# Poll for job status
run.wait_for_completion(show_output=True) # value of True will display a verbose, streaming log

# Examine the recorded metrics from the run
print(run.get_metrics())