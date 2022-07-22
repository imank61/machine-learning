#Practice 1
#Linear Regression & diabetes data set.
#This code uses streamlit to show the results.
#Provided by Iman.Khosrojerdi

#Importing basic packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sns.set()

print('The result will shown on the streamlit\n')

#setting streamlit web page
st.title('Linear Regression & diabetes data set')
st.subheader('provided by Iman.Khosrojerdi')
st.subheader('')
st.subheader('')

progress_bar = st.sidebar.progress(0)
st.sidebar.empty().text('starting...')

#loading data
df = pd.read_csv('diabetes.txt', delimiter='\t')
st.subheader('Loaded data set:')
st.dataframe(df)

#splitting data
x = df.iloc[:,0:10]
y = df.iloc[:,10:]

#Splites data to the train and test parts with 0.8 & 0.2 ratio
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)

#creating the Linear Regression
regr = LinearRegression()

#fitting the classifier
regr = regr.fit(x_train,y_train) 
st.sidebar.empty().text('The model created.')

#Computing the Coefficients and corrolations
progress_bar.progress(30)
coef=regr.coef_
corr=df.corr()
st.sidebar.empty().text('Ceof & corr factors computed.')

#showing Heat map on the streamlit
st.subheader('Heat map of correlation:')
fig, ax = plt.subplots()
sns.heatmap(corr, square=False, annot=True, cmap='RdYlGn', ax=ax)
st.write(fig)

#showing important features on the streamlit
st.subheader('Important features:')
feature_imp = pd.Series(regr.coef_[0], index=x.columns)
st.bar_chart(feature_imp)

#predicting the X_test
progress_bar.progress(70)
y_pred = regr.predict(x_test)
st.sidebar.empty().text('X_test predicted.')

#Showing the results
st.subheader('The results:')
st.write("Mean squared error: {0}".format(round(sm.mean_squared_error(y_test, y_pred), ndigits=2)))
st.write("Mean absolute error: {0}".format(round(sm.mean_absolute_error(y_test, y_pred), ndigits=2)))
st.write("Median absolute error: {0}".format(round(sm.median_absolute_error(y_test, y_pred), ndigits=2)))
st.write("Explained variance score: {0}".format(round(sm.explained_variance_score(y_test, y_pred), ndigits=2)))
st.write("R2 score: {0}".format(round(sm.r2_score(y_test, y_pred), ndigits=2)))

st.sidebar.empty().text('Finished!')
progress_bar.progress(100)







