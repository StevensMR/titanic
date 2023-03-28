# Titanic Survival Predictor
This is my take on a classification model on the Titanic dataset, predicting who survived the sinking of the Titanic.

## Status
  
On 23 Feb 2023, submitted predictions to kaggle Titanic comptition using a sk-learn Gaussian Process Classifier for an accuracy score of 78.229%

28 Mar 2023, I updated the code into an expanded EDA notebook and a streamlined model building and testing notebook, and updated this readme file with images. New hyperparameter tuned XGBoost model scored 78.468%

## Project Description

This is a machine learning classification model written in Python, utilizing SciKit-Learn's models to train a classification model on a training dataset of Titanic passengers to predict the survivors from a test set of Titanic passengers.   

### Files
  
The project is divided into two data files (.csv), two Jupyter notebooks, a pickle file, and a csv submission file

#### Data
  
    train.csv : The training data for the titanic competition, available from https://www.kaggle.com/competitions/titanic/data
    
    test.csv : The test data, also from https://www.kaggle.com/competitions/titanic/data, which the predictor will run against to develop the sumbission file.
    
#### Code
  
    titanic_EDA.ipynb : Exploratory Data Analysis to inform the building of the model.
    
    Titanic_w_teapot : preprocessing, model evaluation, hyperparameter tuning, and prediction on the test set.  Exports submission.csv and Titanic_model.pkl
    
    Titanic_model.pkl : Pickle file of the model, 
    
#### Output
  
    submission.csv : predictions of survivors from the test data

## Execution
Skipping the EDA and the model building, the Titanic_w_teapot.ipynb file can be loaded into a conda environment with test.csv file in same directory, then run all cells in the notebook to generate a submission.csv file and the Titanic_model.pkl file.

## Exploratory Data Analysis

Conducting EDA on the training data set on code borrowed from https://drivendata.co/blog/predict-flu-vaccine-data-benchmark/ produced the charts below, ignoring the Passenger ID, Name, Ticket #, and Cabin features, and putting the continuous values of Fare and Age into bins. 
![Titanic EDA](https://github.com/StevensMR/titanic/blob/main/titanic_EDA.png)

From the charts above, we can see that our target, Survival, correlated with the features as follows:
* Passenger Class (Pclass) - First Class are more likely to survive than 3rd class passengers
* Gender (Sex) - Females are much more likely to survive than males
* Embarkation port (Embarked) - C = Cherbourg, Q = Queenstown, S = Southampton - Cherbourg has higher survival rate than Southampton
* Siblings/Spouses (SibSp) - Higher survival rates for 1 or 2 siblings/spouses but then drops off
* Parents/Children (Parch) - Higher survival rates for small families (2-4 members)
* Age (Age bins) - Children (<18) are more likely to survive, but other features are probably more significant to predict survival (note: of the 714 samples in the train set, 177 do not have age listed (NaN values), so we'll have to fill those values)
* Fare (Fare bins) - Strong correlation between the fare for the trip and survival rates, probably also correlated with Passenger Class

## Model Selection

I created a list of classifiers and ran them through a model evaluation and voting ensemble set of functions, producing the chart below.  Examination of the code in the notebook will show additional models I evaluated and then eliminated (through commenting them out) from the hard and soft voting ensembles.

![Titanic Classifier Model Comparison](https://github.com/StevensMR/titanic/blob/main/Model_comparison.png)

Per the chart above, a hyperparameter tuned version of XGBoost outperformed every other model on the training data, including the ensemble learning models.  TPOT confirmed that XGBoost outperformed the other classifiers (0.793 accuracy score), although the tuned XGBoost model returned higher accuracy (0.844 accuracy on the out-of-sample training set)

### Hyperparameter tuning

I used sklearn BayesSearchCV to search for optimum hyperparameters for XGBoost and compared them to the hyperparameters for XGBoost returned from the TPOT code.

TPOT: XGBClassifier(eta= 0.1, gamma= 0, max_depth= 10, max_leaves= 0, min_child_weight= 7, scale_pos_weight= 1)
![XGB10 confusion matrix](https://github.com/StevensMR/titanic/blob/main/xgb10_cm.png)

BayesSearchCV: XGBClassifier(eta= 0.1, gamma= 0, max_depth= 11, max_leaves= 1, min_child_weight= 6, scale_pos_weight= 1)

![XGB11 confusion matrix](https://github.com/StevensMR/titanic/blob/main/xgb11_cm.png)

## Issues
due to the depricated alias of float in the numpy library, I needed to roll back numpy to version 1.23.5 to avoid errors.


## Credits
All code assembled by Mike Stevens with the following exceptions:
EDA graphs: code modified from https://drivendata.co/blog/predict-flu-vaccine-data-benchmark/
TPOT code derived from https://machinelearningmastery.com/tpot-for-automated-machine-learning-in-python/
Voting ensemble code derived from https://machinelearningmastery.com/voting-ensembles-with-python/
Thanks to Brett Waugh who helped mentor me on my beginning steps on my data science/machine learning journey
