#Practice 3
#Softmax Regression & Iris data set.
#This code uses streamlit to show the results.
#Provided by Iman.Khosrojerdi

#Importing basic packages
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import sklearn.metrics as sm


def HasError(Y_test, Y_predict):
    if Y_test == Y_predict:
        return "No"
    else:
        return "Yes"

def PrintResult(Y_test, Y_predict, Y_prob):
    switcher ={
        0: "Setosa",
        1: "versicolor",
        2: "virginica",
    }

    yTestStr = []
    yPredictStr = []
    errorAry = []

    i=0
    while i<Y_test.size :
        yTestStr.append(switcher.get(Y_test[i], "nothing"))
        yPredictStr.append(switcher.get(Y_test[i], "nothing"))
        errorAry.append(HasError(Y_test[i],Y_predict[i]))
        i+=1

    resultDf = pd.DataFrame({
    'Y_test':yTestStr,
    'Y_predict':yPredictStr,
    'Setosa prob':Y_prob[:,0],
    'versicolor prob':Y_prob[:,1],
    'virginica prob':Y_prob[:,2],
    'Error': errorAry})

    del yTestStr, yPredictStr, errorAry
   
    st.subheader('The result of prediction:')
    st.dataframe(resultDf)


print('The result will shown on the streamlit\n')

#setting streamlit web page
st.title('Softmax Regression & Iris data set')
st.subheader('provided by Iman.Khosrojerdi')
st.subheader('')
st.subheader('')

progress_bar = st.sidebar.progress(0)
st.sidebar.empty().text('starting...')

#loading data
irisDs = datasets.load_iris()
st.subheader('Loaded data set:')
irisDf = pd.DataFrame(data=irisDs.data, columns=irisDs.feature_names)
st.dataframe(irisDf)

#splitting data
X = irisDs.data[:, 0:4]
Y = irisDs.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) # 70% train
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

#showing Heat map on the streamlit
st.subheader('Heat map of correlation:')
fig, ax = plt.subplots()
sns.heatmap(irisDf.corr(), square=False, annot=True, cmap='RdYlGn', ax=ax)
st.write(fig)

#creating the Softmax Regression
sr = LogisticRegression(multi_class='multinomial', # switch to Softmax Regression
                                     solver='lbfgs', # handle multinomial loss, L2 penalty
                                     C=10)

#fitting the model
progress_bar.progress(0)
sr.fit(X_train_s, Y_train)
st.sidebar.empty().text('The model created.')

#predicting the X_test
progress_bar.progress(30)
Y_predict = sr.predict(X_test_s)
progress_bar.progress(50)
Y_prob=sr.predict_proba(X_test_s)
progress_bar.progress(70)
Y_prob = np.round(Y_prob, 2)
st.sidebar.empty().text('Prediction done!')

#showing important features on the streamlit
st.subheader('Important features:')
feature_imp = pd.Series(sr.coef_[0], index=irisDs.feature_names)
st.bar_chart(feature_imp)

#showing the result
PrintResult(Y_test, Y_predict, Y_prob)

#showing accuracy information
st.subheader('Accuracy information:')
st.write("accuracy %.3f" % sm.accuracy_score(Y_test, Y_predict))
st.write("mean squared error: %.3f" % sm.mean_squared_error(Y_test, Y_predict))
st.write("mean absolute error: %.3f" % sm.mean_absolute_error(Y_test, Y_predict))
st.write("median absolute error: %.3f" % sm.median_absolute_error(Y_test, Y_predict))
st.write("explained variance score: %.3f" % sm.explained_variance_score(Y_test, Y_predict))
st.write("R2 score: %.3f" % sm.r2_score(Y_test, Y_predict))

st.sidebar.empty().text('Finished!')
progress_bar.progress(100)






