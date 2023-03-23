# Titanic Survival Predictor
This is my take on a classification model on the Titanic dataset, predicting who survived the sinking of the Titanic.

  Status
  
On 23 Feb 2023, submitted predictions to kaggle Titanic comptition using a sk-learn Gaussian Process Classifier for an accuracy score of 78.229%

  Project Description

This is a machine learning classification model written in Python, utilizing SciKit-Learn's models to train a classification model on a dataset of Titanic passengers to predict the survivors from a second list of Titanic passengers.   

  Files
  
The project is divided into two data files (.csv), three Jupyter notebooks, a pickle file, and a csv submission file

  Data
  
    train.csv : The training data for the titanic competition, available from https://www.kaggle.com/competitions/titanic/data
    
    test.csv : The test data, also from https://www.kaggle.com/competitions/titanic/data, which the predictor will run against to develop the sumbission file.
    
  Code
  
    titanic_EDA.ipynb : Exploratory Data Analysis to inform the building of the model.
    
    titanic_model_building : Preprocess the data, build and evaluate the classification models, exporting the best model at the end into a pickle file.
    
    titanic_predictor.ipynb : imports the pickle file, preprocesses the test data, predicts survivors from the test data, and exports the results into submission.csv
    
    Titanic_model.pkl : Pickle file of the model, 
    
  Output
  
    submission.csv : predictions of survivors from the test data

## Execution
Skipping the EDA and the model building, the titanic_predictor.ipynb file can be loaded into a conda environment with test.csv and Titanc_model.pkl file in same directory, then run all cells in the notebook to generate a submission.csv file.

## Model Selection

![Titanic Classifier Model Comparison](https://github.com/StevensMR/titanic/blob/main/Model_comparison.png)

Per the chart above, a hyperparameter tuned version of XGBoost outperformed every other model on the training data, including the ensemble learning models.  TPOT confirmed that XGBoost outperformed the other classifiers (0.793 accuracy score), although the tuned XGBoost model returned higher accuracy (0.844 accuracy)

## Issues
due to the depricated alias of float in the numpy library, I needed to roll back numpy to version 1.23.5 to avoid errors.

## Credits
All code written by Mike Stevens, with the following exceptions:

TPOT code modified from https://machinelearningmastery.com/tpot-for-automated-machine-learning-in-python


Thanks to Brett Waugh who helped mentor me on my beginning steps on my data science/machine learning journey
