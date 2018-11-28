# Step 1 - Load training data and define model training function
################################################################
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
from azureml.core.model import Model 
import pickle
import json

# Verify AML SDK Installed
# view version history at https://pypi.org/project/azureml-sdk/#history 
print("SDK Version:", azureml.core.VERSION)


# Load our training data set
print("Current working directory is ", os.path.abspath(os.path.curdir))
df_affordability = pd.read_csv('./03-model-deployment/data/UsedCars_Affordability.csv', delimiter=',')
print(df_affordability)

full_X = df_affordability[["Age", "KM"]]
full_Y = df_affordability[["Affordable"]]

# Define a helper method that will train, score and register the classifier using different settings
def train_eval_register_model(ws, experiment_name, model_name, full_X, full_Y,training_set_percentage):

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

    # Serialize the model to a pickle file in the outputs folder
    output_model_path = 'outputs/' + model_name + '.pkl'
    pickle.dump(clf,open(output_model_path,'wb'))
    print('Exported model to ', output_model_path)

    # Serialize the scaler as a pickle file in the same folder as the model
    output_scaler_path = 'outputs/' + 'scaler' + '.pkl'
    pickle.dump(scaler,open(output_scaler_path,'wb'))
    print('Exported scaler to ', output_scaler_path)

    # notice for the model_path, we supply the name of the outputs folder without a trailing slash
    # this will ensure both the model and the scaler get uploaded.
    registered_model = Model.register(model_path='outputs', model_name=model_name, workspace=ws)

    print(registered_model.name, registered_model.id, registered_model.version, sep = '\t')

    run.complete()

    return (registered_model, clf, scaler, score, run)



# Step 2 - Retrieve the AML Workspace and Train a model
#######################################################

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


# Create an experiment, log metrics and register the created model
experiment_name = "Experiment-03-30"
model_name = "usedcarsmodel"
training_set_percentage = 0.50
registered_model, model, scaler, score, run = train_eval_register_model(ws, experiment_name, model_name, full_X, full_Y, training_set_percentage)


# Step 3 - Download the registered model, re-load  the model and verify it still works
######################################################################################
# Download the model to a local directory
model_path = Model.get_model_path(model_name, _workspace=ws)
age = 60
km = 40000

# Re-load the model
scaler = pickle.load(open(os.path.join(model_path,'scaler.pkl'),'rb'))
scaled_input = scaler.transform([[age, km]])
model2 = pickle.load(open(os.path.join(model_path,'usedcarsmodel.pkl'), 'rb'))

# Use the loaded model to make a prediction
prediction = model2.predict(scaled_input)
print(prediction)
prediction_json = json.dumps(prediction.tolist())
print(prediction_json)


# Step 4 - Create a Conda dependencies environment file
#######################################################
from azureml.core.conda_dependencies import CondaDependencies 

mycondaenv = CondaDependencies.create(conda_packages=['scikit-learn','numpy','pandas'])

with open("mydeployenv.yml","w") as f:
    f.write(mycondaenv.serialize_to_string())


# Step 5 - Create container image configuration
###############################################

# Create the scoring script
# See the scoring script available in ./score.py

# Build the ContainerImage
runtime = "python" 
driver_file = "03-model-deployment-score.py"
conda_file = "mydeployenv.yml"

from azureml.core.image import ContainerImage

image_config = ContainerImage.image_configuration(execution_script = driver_file,
                                                  runtime = runtime,
                                                  conda_file = conda_file)


# Step 6 - Create ACI configuration
####################################
from azureml.core.webservice import AciWebservice, Webservice

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1, 
    memory_gb = 1, 
    tags = {'name':'Azure ML ACI'}, 
    description = 'This is a great example.')



# Step 7 -Deploy the webservice to ACI
######################################
service_name = "usedcarsmlservice01"

webservice = Webservice.deploy_from_model(
  workspace=ws, 
  name=service_name, 
  deployment_config=aci_config,
  models = [registered_model], 
  image_config=image_config, 
  )

webservice.wait_for_deployment(show_output=True)


# Step 8 - Test the ACI deployed webservice
###########################################
import json
age = 60
km = 40000
test_data  = json.dumps([[age,km]])
test_data
result = webservice.run(input_data=test_data)
print(result)




# Step 9 - Provision an AKS cluster 
####################################
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice

# Use the default configuration, overriding the default location to a known region that supports AKS
prov_config = AksCompute.provisioning_configuration(location='westus2')

aks_name = 'aks-cluster01' 

# Create the cluster
aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_name, 
                                  provisioning_configuration = prov_config)


# Wait for cluster to be ready
aks_target.wait_for_completion(show_output = True)
print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)


# Step 10 - Deploy webservice to AKS
####################################
# Create the web service configuration (using defaults)
aks_config = AksWebservice.deploy_configuration()

aks_service_name ='usedcarsaksservice'

aks_service = Webservice.deploy_from_model(
  workspace=ws, 
  name=aks_service_name, 
  deployment_config=aks_config,
  models = [registered_model], 
  image_config=image_config,
  deployment_target=aks_target
  )


aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)


# Step 11 - Test the AKS deployed webservice
############################################
import json
age = 60
km = 40000
test_data  = json.dumps([[age,km]])
test_data
result = aks_service.run(input_data=test_data)
print(result)