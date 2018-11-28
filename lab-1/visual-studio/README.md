# Lab 1 - Training a Machine Learning Model using Azure Machine Learning service

In this lab you will setup the Azure Machine Learning service from code and create a classical machine learning model that logs metrics collected during model training.

## Exercise 0 - Get the lab files
If you have not cloned this repository to your working environment, do so now. All of the artifacts for this lab are located under `starter-artifacts/visual-studio`.

## Exercise 1 - Get oriented to the lab files

1. Navigate to the directory where you cloned the repo and then open the solution `azure-ml-labs.sln`. 
2. With the solution open in Visual Studio, look at Solution Explorer and expand the project `01-model-training`.
3. Under that, expand Python Environments. You should see one environment called `AzureML (3.6, 64-bit)`. This Anaconda environment will be used when you execute code either by running the script using F5 or by executing Python code in the Python Interactive Window.
4. Expand the `data` folder. This folder contains two CSV files. `UsedCars_Clean.csv` represents the unlabeled data and `UsedCars_Affordability.csv` contains the complete data set with labels (Affordable is 1 for affordable, 0 for not affordable).
5. Expand `training`. This folder contains train.py which will be used later in the lab to train the model using a remote cluster provided by Azure Batch AI.
6. Open `_01_model_training.py`. This is the Python file you will step thru executing in this lab. Leave it open and continue to the next exercise.


## Exercise 2 - Train a simple model locally
1. Read thru and select the code starting with # Step 1 all the way down to but NOT including # Step 2. Use `Control + Enter` to execute the selected code in the Python Immediate Window. Take a moment to look at the data loaded into the Pandas Dataframe - it contains data about used cars such as the price (in dollars), age (in years), KM (kilometers driven) and other attributes like weather it is automatic transimission, the number of doors, and the weight.
2. In Step 2, we are going to try and build a model that can answer the question "Can I afford a car that is X months old and has Y kilometers on it, given I have $12,000 to spend?". We will engineer the label for affordable. Select the code starting with # Step 2 all the way down to but NOT including # Step 3. Use `Control + Enter` to execute the selected code in the Python Immediate Window.
3. We are going to train a Logistic Regression model locally. This type of model requires us to standardize the scale of our training features Age and KM, so we use the `StandardScaler` from Scikit-Learn to transform these features so that they have values centered with a mean around 0 (mostly between -2.96 and 1.29). Select Step 3 and execute the code. Observe the difference in min and max values between the un-scaled and scaled Dataframes.  
4. Train the model by fitting a LogisticRegression against the scaled input features (X_scaled) and the labels (y). Select Step 4 and execute the code.
5. Try prediction - if you set the age to 60 months and km to 40,000, does the model predict you can afford the car? Execute Step 5 and find out. 
6. Now, let's get a sense for how accurate the model is. Select and execute Step 6. What was your model's accuracy?
7. One thing that can affect the model's performance is how much data of all the labeled training data available is used to train the model. In Step 7, you define a method that uses train_test_split from Scikit-Learn that will enable you to split the data using different percentages. Execute Step 7 to register this function.

## Exercise 3 - Use Azure Machine Learning to log performance metrics
In the steps that follow, you will train multiple models using different sizes of training data and observe the impact on performance (accuracy). Each time you create new model, you are executing a Run in the terminology of Azure Machine Learning service. In this case, you will create one Experiment and execute multiple Runs within it, each with different training percentages (and resultant varying accuracies). 

1. Execute Step 8 to quickly verify you have the Azure Machine Learning SDK installed. If you get a version number back without error, you are ready to proceed.
2. All Azure Machine Learning entities are organized within a Workspace. You can create an AML Workspace in the Azure Portal, but as the code in Step 9 shows, you can also create a Workspace directly from code. Set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments. Execute Step 9. You will be prompted to log in to your Azure Subscription.
3. To begin capturing metrics, you must first create an Experiment and then call `start_logging()` on that Experiment. The return value of this call is a Run. This root run can have other child runs. When you are finished with an experiment run, use `complete()` to close out the root run. Execute Step 10 to train four different models using differing amounts of training data and log the results to Azure Machine Learning.
4. Now that you have captured history for various runs, you can review the runs. You could use the Azure Portal for this - go to the Azure Portal, find your Azure Machine Learning Workspace, select Experiments and select the UsedCars_Experiment. However, in this case we will use the AML SDK to query for the runs. Select and execute Step 11 to view the runs and their status.

## Exercise 4 - Train remotely using Azure Batch AI
Up until now, all of your training was executed locally on the same machine running Visual Studio. Now you will execute the same logic targeting a remote Azure Batch AI cluster, which you will provision from code.

1. Read thru and then execute Step 12 in which you will create an Azure Batch AI cluster using code. Once your cluster is ready, you should see output similar to the following:
```
Creating a new compute target...
Creating
succeeded.....
BatchAI wait for completion finished
Minimum number of nodes requested have been provisioned
{'allocationState': 'steady', 'allocationStateTransitionTime': '2018-11-17T17:56:07.361000+00:00', 'creationTime': '2018-11-17T17:52:53.601000+00:00', 'currentNodeCount': 1, 'errors': None, 'nodeStateCounts': {'idleNodeCount': 0, 'leavingNodeCount': 0, 'preparingNodeCount': 1, 'runningNodeCount': 0, 'unusableNodeCount': 0}, 'provisioningState': 'succeeded', 'provisioningStateTransitionTime': '2018-11-17T17:53:59.653000+00:00', 'scaleSettings': {'manual': None, 'autoScale': {'maximumNodeCount': 3, 'minimumNodeCount': 1, 'initialNodeCount': 1}}, 'vmPriority': 'lowpriority', 'vmSize': 'STANDARD_DS11_V2'}

```
2. With your cluster ready, you need to upload the training data to the default DataStore for your AML Workspace (which uses Azure Storage). Execute Step 13 to upload the data folder.  
3. Next, you will need to create a training script that is similar to the code you have executed locally to train the model. Open `training/train.py` and read thru it. You do not need to execute this script, as you will send it to Azure Batch AI for execution. 
4. Return to `_01_model_training.py`. You will create an estimator that describes the configuration of the job that will execute your model training script. Execute Step 14 to create this estimator.
5. As the last step, submit the job using the `submit()` method of the Experiment object. Execute Step 15 to remotely execute your training script. The output you should see will begin with the creation of a Docker Container that contains your configured dependencies, followed by the execution of your training script.



