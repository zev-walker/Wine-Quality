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
st.sidebar.title("🍷 Wine Quality ML App")
page = st.sidebar.radio("Navigate", ["📊 Data Exploration", "⚙️ Train & Tune", "📈 Evaluation & Prediction"])

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
# 2️⃣ Train & Tune Page
# =====================
elif page == "⚙️ Train & Tune":
    st.header("Train Models")

    model_choice = st.selectbox("Choose Model", ["Random Forest", "SVM"])
    wine_choice = st.selectbox("Choose Wine Type", ["Red", "White"])

    if st.button("Train Model"):
        # Select dataset
        X = red_scaled[features] if wine_choice == "Red" else white_scaled[features]
        y = red_scaled[target] if wine_choice == "Red" else white_scaled[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define model and hyperparameters
        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
        else:  # SVM
            model = SVC(probability=True)
            param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

        # Grid search
        grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Save in session state
        if wine_choice == "Red":
            st.session_state.best_model_red = best_model
        else:
            st.session_state.best_model_white = best_model

        st.success(f"✅ Best {model_choice} model trained for {wine_choice} wine!")
        st.write("Best Parameters:", grid.best_params_)

        # -----------------
        # Predictions & Evaluation
        # -----------------
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write(f"**Accuracy:** {acc:.4f} | **Weighted F1 Score:** {f1:.4f}")

        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = sorted(y_test.unique())
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{wine_choice} Wine Confusion Matrix")
        st.pyplot(fig)

        # -----------------
        # Feature Importance / Coefficients
        # -----------------
        if model_choice == "Random Forest":
            importances = best_model.feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': (importances / importances.sum()) * 100
            }).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
            ax.set_title(f"{wine_choice} Wine Feature Importance (%)")
            for i, v in enumerate(feat_imp['Importance']):
                ax.text(v + 0.5, i, f"{v:.1f}%", color='black', va='center')
            st.pyplot(fig)

        elif model_choice == "SVM" and best_model.kernel == "linear":
            coefs = best_model.coef_[0]
            coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs})
            coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm', ax=ax)
            ax.set_title(f"{wine_choice} Wine SVM Feature Coefficients")
            st.pyplot(fig)
        else:
            st.info("⚠️ Feature importance plot not available for SVM with non-linear kernel.")

# =====================
# 3️⃣ Evaluation & Prediction Page
# =====================
elif page == "📈 Evaluation & Prediction":
    st.header("Evaluate Models & Live Prediction")

    # Cross-validation function
    def cross_validate_model(model, X, y, wine_type):
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        st.subheader(f"{wine_type} Wine Cross-Validation")
        st.write("Fold F1 Scores:", scores)
        st.write("Mean F1 Score:", scores.mean())

    # Evaluate trained models
    if "best_model_red" in st.session_state:
        cross_validate_model(st.session_state.best_model_red, red_scaled[features], red_scaled[target], "Red")

    if "best_model_white" in st.session_state:
        cross_validate_model(st.session_state.best_model_white, white_scaled[features], white_scaled[target], "White")

    # -----------------
    # Live Prediction
    # -----------------
    st.subheader("🔮 Live Quality Prediction")
    wine_choice = st.selectbox("Choose Wine Type", ["Red", "White"])
    feature_values = []
    dataset = red_wine if wine_choice == "Red" else white_wine

    for feat in features:
        val = st.slider(
            f"{feat}",
            min_value=float(dataset[feat].min()),
            max_value=float(dataset[feat].max()),
            value=float(dataset[feat].mean()),
            step=0.01
        )
        feature_values.append(val)

    if st.button("Predict Quality"):
        input_arr = np.array(feature_values).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)
        model = st.session_state.get("best_model_red") if wine_choice == "Red" else st.session_state.get("best_model_white")

        if model is not None:
            pred = model.predict(input_scaled)[0]
            st.success(f"Predicted {wine_choice} Wine Quality: **{pred}**")
        else:
            st.error("Please train the model first on the 'Train & Tune' page.")






