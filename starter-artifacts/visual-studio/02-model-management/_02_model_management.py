# Step 1 - load the training data locally
#########################################
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
import pickle

print("Current working directory is ", os.path.abspath(os.path.curdir))
df_affordability = pd.read_csv('data/UsedCars_Affordability.csv', delimiter=',')
print(df_affordability)

full_X = df_affordability[["Age", "KM"]]
full_Y = df_affordability[["Affordable"]]

# Step 2 - Define a helper method for training, evaluating and registering a model
################################################################################### 
def train_eval_register_model(experiment_name, full_X, full_Y,training_set_percentage):

    # start a training run by defining an experiment
    myexperiment = Experiment(ws, experiment_name)
    run = myexperiment.start_logging()


    train_X, test_X, train_Y, test_Y = train_test_split(full_X, full_Y, train_size=training_set_percentage, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    clf = linear_model.LogisticRegression(C=1)
    clf.fit(X_scaled, train_Y)

    scaled_inputs = scaler.transform(test_X)
    predictions = clf.predict(scaled_inputs)
    score = accuracy_score(test_Y, predictions)

    print("With %0.2f percent of data, model accuracy reached %0.4f." % (training_set_percentage, score))

    # Log the training metrics to Azure Machine Learning service run history
    run.log("Training_Set_Percentage", training_set_percentage)
    run.log("Accuracy", score)
    run.complete()

    output_model_path = 'outputs/' + experiment_name + '.pkl'
    pickle.dump(clf,open(output_model_path,'wb'))

    # Register and upload this version of the model with Azure Machine Learning service
    registered_model = run.register_model(model_name='usedcarsmodel', model_path=output_model_path)

    print(registered_model.name, registered_model.id, registered_model.version, sep = '\t')

    return (clf, score)

# Step 3 - Run a few experiments in your Azure ML Workspace
###########################################################
# Verify AML SDK Installed
print("SDK Version:", azureml.core.VERSION)


# Create a new Workspace or retrieve the existing one
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


# Create an experiment, log metrics and register the created models for multiple training runs
experiment_name = "Experiment-02-01"
training_set_percentage = 0.25
model, score = train_eval_register_model(experiment_name, full_X, full_Y, training_set_percentage)

experiment_name = "Experiment-02-02"
training_set_percentage = 0.5
model, score = train_eval_register_model(experiment_name, full_X, full_Y, training_set_percentage)

experiment_name = "Experiment-02-03"
training_set_percentage = 0.75
model, score = train_eval_register_model(experiment_name, full_X, full_Y, training_set_percentage)


# Step 4 - Query for all Experiments.
#####################################
# You can retreive the list of all experiments in Workspace using the following:
all_experiments = ws.experiments

print(all_experiments)

# Query for the metrics of a particular experiment
# You can retrieve an existing experiment by constructing an Experiment object using the name of an existing experiment.
my_experiment = Experiment(ws, "Experiment-02-03")
print(my_experiment)

# Query an experiment for metrics
# With an experiment in hand, you retrieve any metrics collected for any of its child runs 
my_experiment_runs = my_experiment.get_runs()
print( [ (run.experiment.name, run.id, run.get_metrics()) for run in my_experiment_runs] )



# Step 5 - Submit an experiment to Azure Batch AI and log metrics for multiple training runs
############################################################################################


#ws = Workspace.get(name=workspace_name, subscription_id=subscription_id,resource_group=resource_group)
#print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

experiment_name = "UsedCars_Batch_02"

from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster
batchai_cluster_name = "carscluster02"
cluster_min_nodes = 1
cluster_max_nodes = 3
vm_size = "STANDARD_DS11_V2"
autoscale_enabled = True


if batchai_cluster_name in ws.compute_targets:
    compute_target = ws.compute_targets[batchai_cluster_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('Found existing compute target, using this compute target instead of creating:  ' + batchai_cluster_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,  
                                                                vm_priority = 'lowpriority', # optional
                                                                autoscale_enabled = autoscale_enabled,
                                                                min_nodes = cluster_min_nodes, 
                                                                max_nodes = cluster_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, batchai_cluster_name, provisioning_config)
    
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
     # For a more detailed view of current BatchAI cluster status, use the 'status' property    
    print(compute_target.status.serialize())

# Upload the dataset to the DataStore
ds = ws.get_default_datastore()
print(ds.datastore_type, ds.account_name, ds.container_name)
ds.upload(src_dir='./data', target_path='used_cars', overwrite=True, show_progress=True)


# Prepare batch training script
# - See ./training/train.py


# Create estimator
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
    '--training-set-percentage': 0.3
}

est_config = Estimator(source_directory='./training',
                script_params=script_params,
                compute_target=compute_target,
                entry_script='train.py',
                conda_packages=['scikit-learn','pandas'])

# Execute the job
run = exp.submit(config=est_config)
run

# Poll for job status
run.wait_for_completion(show_output=True) # value of True will display a verbose, streaming log

# Step 6 - Retrieve the metrics for the model trained remotely in Azure Batch AI
################################################################################
# Examine the recorded metrics from the run
print(run.get_metrics())