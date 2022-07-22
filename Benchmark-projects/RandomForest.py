#Final Project
#Random forset & Iris data set.
#This code uses streamlit to show the results.
#Provided by Iman.Khosrojerdi

#Importing basic packages
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import sklearn.metrics as sm

print('The result will shown on the streamlit\n')

#setting streamlit web page
st.title('Random forest & Iris data set')
st.subheader('provided by Iman.Khosrojerdi')
st.subheader('')
st.subheader('')

progress_bar = st.sidebar.progress(0)
st.sidebar.empty().text('starting...')

#loading data
irisDs = datasets.load_iris()
st.subheader('Loaded data set:')
irisDf = pd.DataFrame(data=irisDs.data, columns=irisDs.feature_names)
irisDf['target'] = irisDs['target']
st.dataframe(irisDf)

#splitting data
X = irisDs.data[:, 0:4]
Y = irisDs.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3) # 70% train

#creating the Random forset classifier
clf = RandomForestClassifier(n_estimators=100)

#fitting the classifier
progress_bar.progress(30)
clf.fit(X_train,y_train)
st.sidebar.empty().text('The model created.')

#predicting the X_test
progress_bar.progress(50)
y_pred=clf.predict(X_test)
st.sidebar.empty().text('X_test predicted.')

#showing confusion matrix by streamlit
unique_label = np.unique([y_test, y_pred])
cm = sm.confusion_matrix(y_test, y_pred, labels=unique_label)
index=index=['actual:{:}'.format(x) for x in unique_label]
columns=['prediction:{:}'.format(x) for x in unique_label]
cm_df = pd.DataFrame(cm,index,columns)
st.subheader('Confusion Matrix:')
st.dataframe(cm_df)
progress_bar.progress(70)
st.sidebar.empty().text('Confusion matrix created.')

#showing classificationreports by streamlit
c_report = sm.classification_report(y_test,y_pred,output_dict=True)
df = pd.DataFrame(c_report).transpose()
st.subheader('Classification reports:')
st.dataframe(df)
progress_bar.progress(90)
st.sidebar.empty().text('Classification reports provided.')

#showing Accuracy Score by streamlit
st.subheader("Accuracy Score: {0}".format(round(sm.accuracy_score(y_test,y_pred), ndigits=2)))

st.sidebar.empty().text('Finished!')
progress_bar.progress(100)





