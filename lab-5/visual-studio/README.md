# Lab 5 - Deep Learning

In this lab you train deep learning models built with Keras and a Tensorflow backend that utilize GPUs with the Azure Machine Learning service.

## Exercise 0 - Get the lab files
If you have not cloned this repository to your working environment, do so now. All of the artifacts for this lab are located under `starter-artifacts/visual-studio`.

## Exercise 1 - Get oriented to the lab files

1. Navigate to the directory where you cloned the repo and then open the solution `azure-ml-labs.sln`. 
2. With the solution open in Visual Studio, look at Solution Explorer and expand the project `05-deep-learning`.
3. Under that, expand Python Environments. You should see one environment called `AzureML (3.6, 64-bit)`. This Anaconda environment will be used when you execute code either by running the script using F5 or by executing Python code in the Python Interactive Window.
4. Open `_05_deep_learning.py`. This is the Python file you will step thru executing in this lab. Leave it open and continue to the next exercise.


## Exercise 2 - Train an autoencoder using GPU
1. Begin with Step 1 and read thru the code in Step 1. Here you will use Keras to define an autoencoder. Don't get hung up on the details of constructing the auto-encoder. The point of this lab is to show you how to train neural networks using GPU's. Select Step 1 and type `Control + Enter` to execute the selected code in the Python Immediate Window. In the output, verify that `K.tensorflow_backend._get_available_gpus()` returned an entry describing a GPU available in your environment.
2. Once you have your autoencoder model structured, you need to train the the underlying neural network. Training this model on regular CPU's will take hours. However, you can execute this same code in an environment with GPU's for better performance. Select and execute Step 2. How long did your training take?
3. With a trained auto-encoder in hand, try using the model by selecting and executing Step 3.

## Exercise 3 - Register the neural network model with Azure Machine Learning
1. In Step 4, be sure to set the values for `subscription_id`, `resource_group`, `workspace_name` and `workspace_region` as directed by the comments to create or retrieve your workspace. Observe that you can register a neural network model with Azure Machine Learning in exactly the same way you would register a classical machine learning model. Execute Step 4 to register the model.


 

