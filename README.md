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

##Credits
All code written by Mike Stevens
Thanks to Brett Waugh who helped mentor me on my beginning steps on my data science/machine learning journey
