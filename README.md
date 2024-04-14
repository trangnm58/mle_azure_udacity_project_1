# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
- This dataset contains data about bank marketing campaigns. With this dataset, we try to predict whether a client will subscribe to a term deposit (variable y).
Link to the dataset's description: https://archive.ics.uci.edu/dataset/222/bank+marketing
- The best performing model was a logistic regression model tuned using HyperDrive, achieving an accuracy of 92.08%.

## Scikit-learn Pipeline
- The pipeline architecture includes:
	- Data preprocessing using one-hot encoding and feature engineering for time features.
	- Hyperparameter tuning is performed using a random search method using the Bandit Policy
	- The classification algorithm used is logistic regression.
- Random parameter sampler is efficient in exploring the hyperparameter space randomly. Even though the combinations might not be exhaustive, but it can lead to faster convergence and discovery of good hyperparameter values.
- The benefit of the Bandit early stopping policy is that it terminates poorly performing runs early based on slack factor and evaluation interval, which helps in saving computation resources and time.

## AutoML
- The AutoML generated multiple models including XGBoost, LogisticRegression, SGD, Random Forest, and LightGBM, with various feature engineering techniques and hyperparameters tuned automatically.
- The best performing model was XGBoostClassifier with StandardScalerWrapper as the feature engineering step, the accuracy achieved is 91.68%.

## Pipeline comparison
- The HyperDrive logistic regression model achieved an accuracy of 92.08%, while the AutoML XGBoost model achieved an accuracy of 91.68%.
- Even though XGBoost is a more advanced algorithm than logistic regression, we still observe a slight performance drop with the XGBoost model.
- This difference might be due to the architecture of logistic regression and hyperparamters chosen in the HyperDrive step are more suitable for this particular dataset than the XGBoost.

## Future work
- Some ideas include:
	- Trying out more advanced feature engineering techniques
	- Experimenting with different classification algorithms in the HyperDrive pipeline, or let AutoML run longer.
	- Trying out different hyperparameter sampling techniques in the HyperDrive pipeline
- These improvements might help with selecting a better model, in both the architecture and hyperparameter aspects.