# Step 1 - Create or retrieve your Azure ML Workspace
#####################################################
from azureml.core import Workspace
from azureml.core.model import Model 
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


# Step 2 - Build the ContainerImage for the IoT Edge Module
###########################################################
from azureml.core.image import ContainerImage, Image

runtime = "python" 
driver_file = "iot_score.py"
conda_file = "myenv.yml"

image_config = ContainerImage.image_configuration(execution_script = driver_file,
                                                  runtime = runtime,
                                                  conda_file = conda_file)

model = Model.register(model_path = "model.pkl",
                       model_name = "iot_model.pkl",
                       workspace = ws)

image = Image.create(name = "iotimage",
                     # this is the model object 
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)
image.wait_for_creation(show_output = True)

