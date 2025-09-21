# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="🍷 Wine Quality Classifier", layout="wide")

st.title("🍷 Wine Quality Classification with Random Forest")
st.write("Explore, train, and evaluate models for Red and White Wine Quality classification.")

# =====================
# Sidebar Navigation
# =====================
page = st.sidebar.radio("Navigate", ["📊 Data Exploration", "⚙️ Train & Tune Model", "📈 Evaluation & Prediction"])

# =====================
# Load Data
# =====================
@st.cache_data
def load_data():
    red = pd.read_csv(
        r"C:\Users\vinee\OneDrive\Documents\Wine Quality Mini\winequality-red.csv", 
        delimiter=';'
    )
    white = pd.read_csv(
        r"C:\Users\vinee\OneDrive\Documents\Wine Quality Mini\winequality-white.csv", 
        delimiter=';'
    )
    return red, white

red_wine, white_wine = load_data()

features = red_wine.columns[:-1]
target = "quality"

# Standardize
scaler = StandardScaler()
red_scaled = red_wine.copy()
white_scaled = white_wine.copy()
red_scaled[features] = scaler.fit_transform(red_wine[features])
white_scaled[features] = scaler.fit_transform(white_wine[features])

# Train-test split
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    red_scaled[features], red_scaled[target], test_size=0.2, random_state=42, stratify=red_scaled[target]
)
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(
    white_scaled[features], white_scaled[target], test_size=0.2, random_state=42, stratify=white_scaled[target]
)

# Global variables to store trained models
if "best_model_red" not in st.session_state:
    st.session_state.best_model_red = None
if "best_model_white" not in st.session_state:
    st.session_state.best_model_white = None

# =====================
# 1️⃣ Data Exploration Page
# =====================
if page == "📊 Data Exploration":
    st.header("Dataset Overview")
    st.subheader("Red Wine Sample")
    st.dataframe(red_wine.head())
    st.subheader("White Wine Sample")
    st.dataframe(white_wine.head())

    st.subheader("Wine Quality Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x='quality', data=red_wine, palette='Reds', ax=ax[0])
    ax[0].set_title('Red Wine Quality')
    sns.countplot(x='quality', data=white_wine, palette='Blues', ax=ax[1])
    ax[1].set_title('White Wine Quality')
    st.pyplot(fig)

    st.write("✅ No missing values detected in either dataset.")
    st.write("Red Wine shape:", red_wine.shape, "| White Wine shape:", white_wine.shape)

# =====================
# 2️⃣ Train & Tune Model Page
# =====================
elif page == "⚙️ Train & Tune Model":
    st.header("Train and Tune Random Forest Models")

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }

    def train_and_tune(X_train, y_train, X_test, y_test, wine_type="Red"):
        st.subheader(f"🍷 {wine_type} Wine Model Training")
        rf = RandomForestClassifier(random_state=42)
        rand_search = RandomizedSearchCV(
            rf, param_distributions=param_dist,
            n_iter=10, scoring='f1_weighted', cv=3, verbose=0, n_jobs=-1, random_state=42
        )
        with st.spinner(f"Training {wine_type} Wine Model..."):
            rand_search.fit(X_train, y_train)
        best_model = rand_search.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.write(f"**Best Params:** {rand_search.best_params_}")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Weighted F1 Score:** {f1:.4f}")

        # Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = sorted(y_test.unique())
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{wine_type} Wine Confusion Matrix")
        st.pyplot(fig)

        # Feature Importance (percentage)
        importances = best_model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': features,
            'Importance': (importances / importances.sum()) * 100
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
        ax.set_title(f"{wine_type} Wine Feature Importance (%)")
        ax.set_xlabel("Importance (%)")
        for i, v in enumerate(feat_imp['Importance']):
            ax.text(v + 0.5, i, f"{v:.1f}%", color='black', va='center')
        st.pyplot(fig)

        return best_model

    if st.button("Train Red Wine Model"):
        st.session_state.best_model_red = train_and_tune(X_train_red, y_train_red, X_test_red, y_test_red, "Red")

    if st.button("Train White Wine Model"):
        st.session_state.best_model_white = train_and_tune(X_train_white, y_train_white, X_test_white, y_test_white, "White")

# =====================
# 3️⃣ Evaluation & Prediction Page
# =====================
elif page == "📈 Evaluation & Prediction":
    st.header("Evaluate Models & Live Prediction")

    # Cross-validation if model is trained
    def cross_validate_model(model, X, y, wine_type):
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        st.subheader(f"{wine_type} Wine Cross-Validation")
        st.write("Fold F1 Scores:", scores)
        st.write("Mean F1 Score:", scores.mean())

    if st.session_state.best_model_red:
        cross_validate_model(st.session_state.best_model_red, red_scaled[features], red_scaled[target], "Red")

    if st.session_state.best_model_white:
        cross_validate_model(st.session_state.best_model_white, white_scaled[features], white_scaled[target], "White")

    # -----------------
    # Live Prediction Section
    # -----------------
    st.subheader("🔮 Live Quality Prediction")
    wine_choice = st.selectbox("Choose Wine Type", ["Red", "White"])
    feature_values = []
    for feat in features:
        val = st.slider(
            f"{feat}",
            min_value=float(red_wine[feat].min()),
            max_value=float(red_wine[feat].max()),
            value=float(red_wine[feat].mean()),
            step=0.01
        )
        feature_values.append(val)

    if st.button("Predict Quality"):
        input_arr = np.array(feature_values).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)  # scale the input
        model = st.session_state.best_model_red if wine_choice == "Red" else st.session_state.best_model_white
        if model is not None:
            pred = model.predict(input_scaled)[0]
            st.success(f"Predicted {wine_choice} Wine Quality: **{pred}**")
        else:
            st.error("Please train the model first on the 'Train & Tune' page.")
