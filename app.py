import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

# make containers

header = st. container()
datasets = st. container()
features = st. container()
model_training = st. container()


with header:
    st.title("Titanic survival prediction App")
    st.text("In the project we will work with titanic data.")

with datasets:
    st.header('Header1')
    st.text('dataset')
    df=sns.load_dataset('titanic')
    df= df.dropna()
    st.write(df.head())
    st.subheader('Sex details')
    st.bar_chart(df['sex'].value_counts())
    st.subheader('Class details')
    st.bar_chart(df['class'].value_counts())
    st.subheader('Age')
    st.bar_chart(df['age'].value_counts())

with features:
    st.header('header2')
    st.text('features')
    st.markdown("1. ***Feature 1:*** ")
    st.markdown("1. ***Feature 2:*** ")
    st.markdown("1. ***Feature 3:*** ")



with model_training:
    st.header('Header3')
    st.text('model_training')

    #Making columns Featers / Display

    input,display = st.columns(2)

    max_depth=input.slider("How many people do you know",min_value=1,max_value=100,value=50,step=1
)

n_estimators = input.selectbox("How many trees should be there in  a RF", options=[50,100,200,300,"No limit"])

input.write(df.columns)

input_features=input.text_input('Which feature should be used?')

#model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

if n_estimators == "No limit":
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)


X= df[[input_features]]
y= df[['fare']]

model.fit(X,y)
pred = model.predict(y)

display.subheader("mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("mean squared error of the model is: ")
display.write(mean_squared_error(y,pred))
display.subheader("r squared score of the model is: ")
display.write(r2_score(y,pred))
