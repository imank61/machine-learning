#Final Project
#Pycaret & Iris data set.
#Provided by Iman.Khosrojerdi

#Importing basic packages
import pandas as pd  
from pycaret.classification import *
from sklearn import datasets

#loading data
irisDs = datasets.load_iris()
irisDf = pd.DataFrame(data=irisDs.data, columns=irisDs.feature_names)
irisDf['target'] = irisDs['target']

#splitting data
data = irisDf.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = irisDf.drop(data.index).reset_index(drop=True)

#creating setup for pycaret
exp = setup(data = data, target = 'target', session_id=77 )

#comparing models for the best choice
bestModel = compare_models()

#showing the best model
plot_model(bestModel)
evaluate_model(bestModel)

#selecting Logistis rgeression because it is the best result of the comparison.
lr_model = create_model('lr', fold = 10)

#ploting the created model
plot_model(lr_model, plot = 'auc')

#plotting the confusion matrix of the created model
plot_model(lr_model, plot = 'confusion_matrix')

#Predicting the accuracy using the test dataset
predict_model(lr_model)

#Checking with the unseen data
new_prediction = predict_model(lr_model, data=data_unseen)
new_prediction