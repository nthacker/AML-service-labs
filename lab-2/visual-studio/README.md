# Lab 2 - Using Azure Machine Learning service Model Versioning and Run History

In this lab you will use the capabilities of the Azure Machine Learning service to collect model performance metrics and to capture model version, as well as query the experimentation run history to retrieve captured metrics. 

## Exercise 0 - Get the lab files
If you have not cloned this repository to your working environment, do so now. All of the artifacts for this lab are located under `starter-artifacts/visual-studio`.

## Exercise 1 - Get oriented to the lab files

1. Navigate to the directory where you cloned the repo and then open the solution `azure-ml-labs.sln`. 
2. With the solution open in Visual Studio, look at Solution Explorer and expand the project `02-model-management`.
3. Under that, expand Python Environments. You should see one environment called `AzureML (3.6, 64-bit)`. This Anaconda environment will be used when you execute code either by running the script using F5 or by executing Python code in the Python Interactive Window.
4. Expand the `data` folder. This folder contains one CSV files. `UsedCars_Affordability.csv` contains the complete, cleaned data set with labels (Affordable is 1 for affordable, 0 for not affordable).
5. Expand `training`. This folder contains train.py which will be used later in the lab to train the model using a remote cluster provided by Azure Batch AI.
6. Open `_02_model_management.py`. This is the Python file you will step thru executing in this lab. Leave it open and continue to the next exercise.


## Exercise 2 - Train a simple model locally
This lab builds upon the lessons learned in the previous lab, but is self contained so you work thru this lab without having to run a previous lab. As such Steps 1, 2 and 3 in the lab are not explored in detail as their goal is to setup a few experiment runs, which was covered in detail in Lab 1.
1. Read thru and select the code starting with # Step 1 all the way down to but NOT including # Step 2. Use `Control + Enter` to execute the selected code in the Python Immediate Window. Take a moment to look at the data loaded into the Pandas Dataframe - it contains data about used cars such as the price (in dollars), age (in years), KM (kilometers driven) and other attributes like weather it is automatic transimission, the number of doors, and the weight.
2. In Step 2, we will define a helper method that locally trains, evaluates and then registers the trained model with Azure Machine Learning. Select and execute Step #2.
3. In Step 3, we retrieve an existing Azure Machine Learning Workspace (or create a new one if desired). In this step, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments. With the Workspace retrieved, we will train 3 different models using different subsets of the training data. Select and execute Step #3.


## Exercise 3 - Use Azure Machine Learning to query for performance metrics
1. As was demonstrated in the previous lab, you can use the Workspace to get a list of Experiments. You can also query for a particular Experiment by name. With an Experiment in hand, you review all runs associated with that Experiment and retrieve the metrics associated with each run. Select and execute Step #4 to see this process. What was the accuracy of the only run for Experiment-02-03?


## Exercise 4 - Remotely train a model in Azure Batch AI
1. Remote model training was covered in the previous lab. Execute Step #5 to create or retreive your Azure Batch AI cluster and the submit to it a model training job. Wait for the run to complete before proceeding to the next exercise.

## Exercise 5 - Retrieve metrics for the remote Run
1. You can easily retrieve the metrics for a Run executed remotely by using `run` object returned by the call to `Experiment.submit`. Execute Step 6 to retrieve metrics for the run you just executed. What was the accuracy of the run?



