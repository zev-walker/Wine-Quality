# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =====================
# Load Data
# =====================
@st.cache_data
def load_data():
    red = pd.read_csv("winequality-red.csv", delimiter=";")
    white = pd.read_csv("winequality-white.csv", delimiter=";")
    return red, white

red_wine, white_wine = load_data()

# Features and target
features = [col for col in red_wine.columns if col != "quality"]
target = "quality"

# =====================
# Standardize
# =====================
scaler = StandardScaler()
red_scaled = red_wine.copy()
white_scaled = white_wine.copy()
red_scaled[features] = scaler.fit_transform(red_wine[features])
white_scaled[features] = scaler.fit_transform(white_wine[features])

# =====================
# Streamlit Sidebar
# =====================
st.sidebar.title("🍷 Wine Quality ML




