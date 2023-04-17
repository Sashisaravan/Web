import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, Booster, DMatrix
import matplotlib.pyplot as plt

# Load the pre-trained models
# Load the pre-trained XGBoost model
xgboost = Booster(model_file='xgboost.model')
sgboost = pickle.load(open('sgboost.pkl', 'rb'))
DT = pickle.load(open('dt_model.pkl', 'rb'))
KNN = pickle.load(open('knn_model.pkl', 'rb'))
MLP = pickle.load(open('mlp_model.pkl', 'rb'))

# Define the function to make predictions and display the report
def predict(model, data):
    if isinstance(model, Booster): # XG Boost
        dmatrix = DMatrix(data)
        pred = model.predict(dmatrix)

        danger_level = "low" if pred < 0.5 else "high"
        result = "diabetic" if pred >= 0.5 else "not diabetic"
        prob = "1.0" if pred >= 0.5 else "0.0"

        return result, danger_level, pred[0],prob

    elif isinstance(model, DecisionTreeClassifier): # Decision Tree
        pred = model.predict(data)
        proba = model.predict_proba(data)[:, 1]
        result = ""
        danger_level = ""
        if pred[0] == 0:
            result = "Non-diabetic"
            danger_level = "Low"
        else:
            result = "Diabetic"
            if proba[0] < 0.5:
                danger_level = "Medium"
            else:
                danger_level = "High"
        return result, danger_level, proba[0]

    elif isinstance(model, KNeighborsClassifier): # KNN
        pred = model.predict(data)
        proba = model.predict_proba(data)[:, 1]
        result = ""
        danger_level = ""
        if pred[0] == 0:
            result = "Non-diabetic"
            danger_level = "Low"
        else:
            result = "Diabetic"
            if proba[0] < 0.5:
                danger_level = "Medium"
            else:
                danger_level = "High"
        return result, danger_level, proba[0]
    elif isinstance(model, MLPClassifier): # MLP
        pred = model.predict(data)
        proba = model.predict_proba(data)[:, 1]
        result = ""
        danger_level = ""
        if pred[0] == 0:
            result = "Non-diabetic"
            danger_level = "Low"
        else:
            result = "Diabetic"
            if proba[0] < 0.5:
                danger_level = "Medium"
            else:
                danger_level = "High"
        return result, danger_level, proba[0]
    elif isinstance(model, GradientBoostingClassifier): # SGB
        pred = model.predict(data)
        proba = model.predict_proba(data)[:, 1]
        result = ""
        danger_level = ""
        if pred[0] == 0:
            result = "Non-diabetic"
            danger_level = "Low"
        else:
            result = "Diabetic"
            if proba[0] < 0.5:
                danger_level = "Medium"
            else:
                danger_level = "High"
        return result, danger_level, proba[0]
    else:
        return "Error: Invalid Model", "", 0



# Define the function to plot the graph
def plot_wave_graph(proba, danger_level):
    fig, ax = plt.subplots()
    t = np.linspace(0, 4*np.pi, 100)
    wave = np.sin(t) * proba
    ax.plot(t, wave, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability of being diabeteic')
    ax.set_ylim([-1, 1])
    ax.set_title('Diabetes Danger Level')
    st.pyplot(fig)

# Set up the web app interface
st.title("Diabetes Prediction")
st.write("Please fill in the following details to make a prediction.")

Gender = st.selectbox("Gender", ("Male", "Female"))
Pregnancies = st.slider("Pregnancies", 0, 17, 1)
Glucose = st.slider("Glucose", 0, 199, 120)
BloodPressure = st.slider("Blood Pressure", 0, 122, 70)
SkinThickness = st.slider("Skin Thickness", 0, 99, 20)
Insulin = st.slider("Insulin", 0, 846, 79)
BMI = st.slider("BMI", 0.0, 67.1, 26.4)
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47)
Age = st.slider("Age", 21, 81, 33)


# Convert the gender to numerical value
if Gender == "Male":
    Gender = 0
else:
    Gender = 1

# Create a numpy array of the input values
input_data = np.array([[Gender,Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Define the model selection drop-down menu
models = ["XG Boost","SG Boost","Decision Tree","K-nearest neighbor","Multilayer perceptron"]
selected_model = st.selectbox("Select a model", models)

# Define the predict button
if st.button("Predict"):
    if selected_model == "XG Boost":
        result, danger_level, proba, prob= predict(xgboost, input_data)
        st.write("Selected model: XG Boost")
        st.write("Result: The person is", result, "with a", danger_level, "level of danger.")
        st.write("Performance score (accuracy):", proba)
        st.write("Probability of being diabetic:", prob)
        plot_wave_graph(proba, danger_level)

    elif selected_model == "SG Boost":
        result, danger_level, proba = predict(sgboost, input_data)
        accuracy = accuracy_score(sgboost.predict(input_data), [1])
        st.write("Selected model: SG Boost")
        st.write("Result: The person is", result, "with a", danger_level, "level of danger.")
        st.write("Performance score (accuracy):", proba)
        st.write("Probability of being diabetic:", accuracy)
        plot_wave_graph(proba, danger_level)
    elif selected_model == "Decision Tree":
        result, danger_level, proba = predict(DT, input_data)
        accuracy = accuracy_score(DT.predict(input_data), [1])
        st.write("Selected model is:  Decision Tree")
        st.write("Result: The person is", result, "with a", danger_level, "level of danger.")
        st.write("Performance score (accuracy):", proba)
        st.write("Probability of being diabetic:", accuracy)
        plot_wave_graph(proba, danger_level)
    elif selected_model == "K-nearest neighbor":
        result, danger_level, proba = predict(KNN, input_data)
        accuracy = accuracy_score(KNN.predict(input_data), [1])
        st.write("Selected model is: K-nearest neighbor")
        st.write("Result: The person is", result, "with a", danger_level, "level of danger.")
        st.write("Performance score (accuracy):", proba)
        st.write("Probability of being diabetic:", accuracy)
        plot_wave_graph(proba, danger_level)
    else:
        selected_model == "Multilayer perceptron"
        result, danger_level, proba = predict(MLP, input_data)
        accuracy = accuracy_score(MLP.predict(input_data), [1])
        st.write("Selected model is:  Multilayer perceptron")
        st.write("Result: The person is", result, "with a", danger_level, "level of danger.")
        st.write("Performance score (accuracy):", proba)
        st.write("Probability of being diabetic:", accuracy)
        plot_wave_graph(proba, danger_level)