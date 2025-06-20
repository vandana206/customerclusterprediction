# Importing libraries
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

# Load the trained model
kmeans = joblib.load("Model.pkl")

# Load the dataset
df = pd.read_csv("Mall_Customers1.csv")
x = df[["Annual Income (k$)", "Spending Score (1-100)"]]
x_Array = x.values

# Streamlit UI
st.set_page_config(page_title="Customer Cluster Prediction", layout="centered")
st.title("Customer Cluster Prediction")
st.write("Enter the Customer's Annual Income and Spending Score to predict the cluster")

# Input
annual_income = st.number_input("Annual Income of a Customer", min_value=0, max_value=400, value=50)
spending_score = st.slider("Spending Score between 1-100", 1, 100, 20)

# Predict the cluster
if st.button("Predict Cluster"):
    input_data = np.array([[annual_income, spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"The predicted cluster for the customer is: **Cluster {cluster}**")
