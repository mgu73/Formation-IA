import streamlit as st
import pandas as pd
import pickle


st.write("""
# MSDE4 : ML Course
## Iris Flower Prediction App

This app predicts the **Iris flower** type
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 5.0, 3.0)
    petal_length = st.sidebar.slider('Petal length', 1.0, 7.0, 2.0)
    petal_width = st.sidebar.slider('Petal width', 0.1, 3.0, 0.5)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model_iris=pickle.load(open("model_iris.pkl", "rb"))
prediction = model_iris.predict(df)
prediction_proba = model_iris.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(model_iris.classes_))

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

