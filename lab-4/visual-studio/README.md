# Lab 4 - Model Training with AutoML

In this lab you will us the automated machine learning (Auto ML) capabilities within the Azure Machine Learning service to automatically train multiple models with varying algorithms and hyperparameters, select the best performing model and register that model.

## Exercise 0 - Get the lab files
If you have not cloned this repository to your working environment, do so now. All of the artifacts for this lab are located under `starter-artifacts/visual-studio`.

## Exercise 1 - Get oriented to the lab files

1. Navigate to the directory where you cloned the repo and then open the solution `azure-ml-labs.sln`. 
2. With the solution open in Visual Studio, look at Solution Explorer and expand the project `04-automl`.
3. Under that, expand Python Environments. You should see one environment called `AzureML (3.6, 64-bit)`. This Anaconda environment will be used when you execute code either by running the script using F5 or by executing Python code in the Python Interactive Window.
4. Expand the `data` folder. This folder contains one CSV files. `UsedCars_Affordability.csv` contains the complete, cleaned data set with labels (Affordable is 1 for affordable, 0 for not affordable).
5. Open `_04_automl.py`. This is the Python file you will step thru executing in this lab. Leave it open and continue to the next exercise.


## Exercise 2 - Train a model using AutoML
This lab builds upon the lessons learned in the previous lab, but is self contained so you work thru this lab without having to run a previous lab.  
1. Begin with Step 1. In this step you are loading the data prepared in previous labs and acquiring (or creating) an instance of your Azure Machine Learning Workspace. In this step, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments. Read thru the code starting with # Step 1 all the way down to but NOT including # Step 2. Use `Control + Enter` to execute the selected code in the Python Immediate Window.
2. To train a model using AutoML you need only provide a configuration for AutoML that defines items such as the type of model (classification or regression), the performance metric to optimize, exit criteria in terms of max training time and iterations and desired performance, any algorithms that should not be used, and the path into which to output the results. This configuration is specified using the `AutomMLConfig` class, which is then used to drive the submission of an experiment via `experiment.submit`.  When AutoML finishes the parent run, you can easily get the best performing run and model from the returned run object by using `run.get_output()`. Select and execute Step 2 to define the helper function that wraps the AutoML job submission.
3. In Step 3, you invoke the AutoML job. Select and execute Step 3.
4. Try out the best model by using Step 4.

## Exercise 3 - Register an AutoML created model
1. You can register models created by AutoML with Azure Machine Learning just as you would any other model. Select and execute Step 5 to register this model.

 

