# expression_prediction_AI

### Description

This repository contains a project to study and predict the transcriptional state of genes based on the three-dimensional structure of chromatin regions encompassing them. The physical information about chromatin is derived from published datasets of chromatin tracing experiments. Specifically, the DNA MERFISH dataset from Su et al. (2020) and the ORCA dataset from Mateo et al. (2019) are analyzed.

### Notebook description

#### Dataset exploration

The dataset exploration notebooks contain the code necessary to transform the chromatin tracing datasets as given in the original publications into pairwise distances that can be used to train the models, either be it in tabular format or as images for convolutional neural networks. This also includes filtering incomplete traces and faulty imaged points, as detailed in the Oligomodeling publication (currently under review). After publication, I will update this repository to include the fully processed data, currently kept private for confidentiality reasons.
Included notebooks:
1. dataset_exploration_Mateo_2019.ipynb
2. dataset_exploration_Su_2020.ipynb

#### Dataset preprocessing

This notebook contains the code for the preprocessing of the datasets (in the form of arrays) for CNN model training. It makes use of functions in the functions/ module to create test, training, and validation splits required for 5-fold cross validation. Used to preprocess the DNA MERFISH data from Su et al. (2020), the simulated structures derived from it using Oligomodeling (confidential until publication) and the ORCA data from Mateo et al. (2019).
Included notebooks:
1. data_preprocessing.ipynb

#### Model training

1. train_CNN.ipynb: code for the training and overfitting testing of a CNN, originally based on the work by Rajpurkar et al. (2021), and improved by optimizing the network parameters. Using experimental and simulated DNA MERFISH structures as well as ORCA traces.
2. Notebooks to train ML models on the Su et al. (2020) dataset:
    1. train_logistic_regression.ipynb: Classify DNA MERFISH traces using a logistic regression.
    2. train_random_forest.ipynb: Classify traces using a random forest.
    3. train_xgboost.ipynb: Classify traces using XGBoost
3. train_random_forest_Mateo.ipynb: Classify ORCa traces using a random forest.

In all cases I'm interested in the data from Su et al. (2020), but I'm also testing the Mateo et al. (2019) for validation on an already studied dataset. Used 5-fold cross validation for all trainings and a parameter grid search testing up to ~12000 model configurations for the logistic regression, random forest and XGBoost cases.

Managed to match the published results using a much simpler model (random forest instead of CNN). Additionally, my filtering and CNN modifications greatly improved the results shown in the literature, achieving an ROC AUC of around 0.75 and much more balanced results in terms of precision and recall.

#### Environment
Using conda to create an environment including pytorch and scikit-learn (including many packages).