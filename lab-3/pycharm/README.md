# Lab 3 - Model Deployment using Azure Machine Learning service

In this lab you will deploy a trained model to containers using an Azure Container Instance and and Azure Kubernetes Service using the Azure Machine Learning SDK. 

## Exercise 0 - Get the lab files
If you have not cloned this repository to your working environment, do so now. All of the artifacts for this lab are located under `starter-artifacts/pycharm`.

## Exercise 1 - Get oriented to the lab files
1. Within PyCharm, select open Existing Project and navigate to the directory where you cloned the repo to open the project `azure-ml-labs`. 
2. In the Project window, expand External Libraries. You should see one environment called `<Python 3.6>` where the path points to your AzureML Anaconda environment. This Anaconda environment will be used when you execute code.
3. In the Project tool window expand the folder `03-model-deployment`.
4. Expand the `data` folder. This folder contains the CSV file `UsedCars_Affordability.csv` which contains the complete data set with labels (Affordable is 1 for affordable, 0 for not affordable).
5. Open `03_model_deployment.py`. This is the Python file you will step thru executing in this lab. Leave it open and continue to the next exercise.


## Exercise 2 - Train a simple model locally
This lab builds upon the lessons learned in the previous lab, but is self contained so you work thru this lab without having to run a previous lab.  
1. Read thru and select the code starting with # Step 1 all the way down to but NOT including # Step 2. Use `Alt + Shift + Enter` to execute the selected code in the Python Immediate Window. Take a moment to look at the data loaded into the Pandas Dataframe - it contains data about used cars such as the price (in dollars), age (in years), KM (kilometers driven) and other attributes like weather it is automatic transimission, the number of doors, and the weight. In the function `train_eval_register_model` observe how the trained model is saved to the ./outputs folder along with the scaler that will be needed to scale inputs used later when scoring. Observe that we use `Model.register` to upload all files in the ./outputs folder to Azure Machine Learning as the model files. These model files will be retrieved later when the model is deployed into a container and operationalized as a web service.
2. In Step 2, we retrieve or create the AML Workspace and then train one instance of the model that we will deploy. In this step, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments. Select and execute Step #2.


## Exercise 3 - Download a version of a model from Azure Machine Learning
1. Once a model is registered with Azure Machine Learning, we can download the model files to any client and use them for scoring. In Step 3, you download the model you just registered, load both the scaler and model files retrieved by deserializing them into objects and then use them to perform a single prediction. Select and execute Step 3.


## Exercise 4 - Create the container image configuration
1. When you deploy a model as web service to either ACI or AKS, you are deploying a Docker container. The first steps towards deploying involve defining the contents of that container. In Step 4, you create Conda Dependencies YAML file that describes what Python packages need to be installed in the container- in this case you specify scikit-learn, numpy and pandas. Select and execute Step 4.
2. With Azure Machine Learning, you have full control over the logic of the webservice which includes how it loads your model, transforms web service inputs, uses the model for scoring and returns the result. From the Project window, open `03-model-deployment-score.py` and read thru the code that defines the webservice. You do not need to execute this code as the file will be deployed in the contents of the container image you are about to create.
3. Return to `03_model_deployment.py`. To create a Container Image, you need three things: the scoring script file, the runtime configuration (defining whether Python or PySpark should be used) and the Conda Dependencies file. Calling `ContainerImage.image_configuration` will capture all of the container image configuration in a single object. Select and execute Step 5.

## Exercise 5 - Deploy the container image to ACI
1. With the Container Image configuration in hand, you are almost ready to deploy to ACI. The next step is to define the size of the VM that ACI will use to run your Container. Execute Step 6 to create this configuration.
2. To deploy the container that operationalizes your model as a webservice, you can use `Webservice.deploy_from_model` which will use your registered model, and automate the creation of a new Container Image, and run the created container in ACI. Execute Step 7 to deploy your webservice to ACI. This step will take 5-7 minutes to complete.
3. Once the webservice deployment completes, you can use the returned webservice object to invoke the webservice. Execute Step 8 to invoke your webservice deployed to ACI.
 
## Exercise 6 - Deploy the container image to AKS
1. Once you are familiar with the process for deploying a webservice to ACI, you will find the process for deploying to AKS to be similar with one additional step that creates the AKS cluster first. Execute Step 9 to provision a small AKS cluster. This step will take about 15-20 minutes.
2. With your AKS cluster ready, now you can deploy your webservice. Once again, you need to provide a configuration for the size of resources allocated from the AKS cluster to run instances of your Container. Execute Step 10 to deploy your webservice. This step will take 5-7 minutes.
3. As before, you can use the webservice object returned by the deploy_from_model method to invoke your deployed webservice. Execute Step 11 to verify you can invoke the web service.


