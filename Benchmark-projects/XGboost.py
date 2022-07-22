#Final Project
#XGBoost & pima indians diabetes data set.
#This code uses streamlit to show the results.
#Provided by Iman.Khosrojerdi

#Importing basic packages
import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

print('The result will shown on the streamlit\n')

#setting streamlit web page
st.title('XGboost & pima indians diabetes data set')
st.subheader('provided by Iman.Khosrojerdi')
st.subheader('')
st.subheader('')

progress_bar = st.sidebar.progress(0)
st.sidebar.empty().text('starting...')

#loading data
diabetesDf = pd.read_csv('pima-indians-diabetes.csv')
st.subheader('Loaded data set:')
st.dataframe(diabetesDf)

#splitting data
X = diabetesDf.drop(['Outcome'], axis=1)
Y = diabetesDf['Outcome']
X_trian, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#creating the XGboost classifier
xgbc = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)

#fitting the classifier
progress_bar.progress(30)
xgbc.fit(X_trian,y_train)
st.sidebar.empty().text('The model created.')

#predicting the X_test
progress_bar.progress(50)
prediction_xgbc=xgbc.predict(X_test)
st.sidebar.empty().text('X_test predicted.')

#showing confusion matrix by streamlit
unique_label = np.unique([y_test, prediction_xgbc])
cm = confusion_matrix(y_test, prediction_xgbc, labels=unique_label)
index=index=['actual:{:}'.format(x) for x in unique_label]
columns=['prediction:{:}'.format(x) for x in unique_label]
cm_df = pd.DataFrame(cm,index,columns)
st.subheader('Confusion Matrix:')
st.dataframe(cm_df)
progress_bar.progress(70)
st.sidebar.empty().text('Confusion matrix created.')

#showing classification reports by streamlit
c_report=classification_report(y_test,prediction_xgbc,output_dict=True)
df = pd.DataFrame(c_report).transpose()
st.subheader('Classification reports:')
st.dataframe(df)
progress_bar.progress(90)
st.sidebar.empty().text('Classification reports provided.')

#showing Accuracy Score by streamlit
st.subheader("Accuracy Score: {0}".format(round(accuracy_score(y_test,prediction_xgbc), ndigits=2)))

st.sidebar.empty().text('Finished!')
progress_bar.progress(100)

