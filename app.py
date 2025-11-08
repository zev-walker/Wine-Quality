# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score

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
# Streamlit Sidebar
# =====================
st.sidebar.title("🍷 Wine Quality ML App")
page = st.sidebar.radio("Navigate", ["📊 Data Exploration", "⚙️ Train & Tune", "📈 Evaluation & Prediction", "🔍 Model Comparison"])

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
        dataset = red_wine if wine_choice == "Red" else white_wine
        X = dataset[features]
        y = dataset[target]

        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Define model and hyperparameters
        with st.spinner(f"🔄 Training {model_choice} model for {wine_choice} wine..."):
            if model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42)
                param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
            else:  # SVM
                model = SVC(probability=True)
                param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

            # Grid search
            grid = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

        # Save model AND scaler in session state
        model_key = f"best_model_{wine_choice.lower()}"
        scaler_key = f"scaler_{wine_choice.lower()}"
        metrics_key = f"metrics_{wine_choice.lower()}"
        
        st.session_state[model_key] = best_model
        st.session_state[scaler_key] = scaler
        st.session_state[f"model_type_{wine_choice.lower()}"] = model_choice

        st.success(f"✅ Best {model_choice} model trained for {wine_choice} wine!")
        st.write("**Best Parameters:**", grid.best_params_)

        # -----------------
        # Predictions & Evaluation
        # -----------------
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Store metrics
        st.session_state[metrics_key] = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'model_name': model_choice
        }

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")

        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score']))

        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = sorted(y_test.unique())
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{wine_choice} Wine Confusion Matrix")
        st.pyplot(fig)

        # -----------------
        # Feature Importance / Coefficients
        # -----------------
        if model_choice == "Random Forest":
            st.subheader("📊 Feature Importance")
            importances = best_model.feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': (importances / importances.sum()) * 100
            }).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax)
            ax.set_title(f"{wine_choice} Wine Feature Importance (%)")
            for i, v in enumerate(feat_imp['Importance']):
                ax.text(v + 0.5, i, f"{v:.1f}%", color='black', va='center')
            st.pyplot(fig)

        elif model_choice == "SVM" and best_model.kernel == "linear":
            st.subheader("📊 Feature Coefficients")
            coefs = best_model.coef_[0]
            coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs})
            coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm', ax=ax)
            ax.set_title(f"{wine_choice} Wine SVM Feature Coefficients")
            st.pyplot(fig)
        else:
            st.info("⚠️ Feature importance plot not available for SVM with non-linear kernel.")

        # -----------------
        # Download Model
        # -----------------
        st.subheader("💾 Download Trained Model")
        
        # Create a package with model and scaler
        model_package = {
            'model': best_model,
            'scaler': scaler,
            'features': features,
            'model_type': model_choice,
            'wine_type': wine_choice,
            'best_params': grid.best_params_,
            'metrics': {
                'accuracy': acc,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
        }
        
        # Serialize to bytes
        buffer = BytesIO()
        pickle.dump(model_package, buffer)
        buffer.seek(0)
        
        st.download_button(
            label=f"📥 Download {wine_choice} Wine {model_choice} Model",
            data=buffer,
            file_name=f"{wine_choice.lower()}_wine_{model_choice.replace(' ', '_').lower()}_model.pkl",
            mime="application/octet-stream"
        )

# =====================
# 3️⃣ Evaluation & Prediction Page
# =====================
elif page == "📈 Evaluation & Prediction":
    st.header("Evaluate Models & Live Prediction")

    # Cross-validation function
    def cross_validate_model(model, scaler, dataset, wine_type):
        X = dataset[features]
        y = dataset[target]
        X_scaled = scaler.transform(X)
        
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1_weighted')
        st.subheader(f"{wine_type} Wine Cross-Validation Results")
        
        col1, col2 = st.columns(2)
        col1.metric("Mean F1 Score", f"{scores.mean():.4f}")
        col2.metric("Std Deviation", f"{scores.std():.4f}")
        
        # Plot CV scores
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, 6), scores, color='steelblue', alpha=0.7)
        ax.axhline(y=scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.4f}')
        ax.set_xlabel("Fold")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"{wine_type} Wine 5-Fold Cross-Validation")
        ax.legend()
        st.pyplot(fig)

    # Evaluate trained models
    if "best_model_red" in st.session_state and "scaler_red" in st.session_state:
        cross_validate_model(
            st.session_state.best_model_red, 
            st.session_state.scaler_red,
            red_wine,
            "Red"
        )
    else:
        st.info("ℹ️ Train the Red wine model first on the 'Train & Tune' page.")

    if "best_model_white" in st.session_state and "scaler_white" in st.session_state:
        cross_validate_model(
            st.session_state.best_model_white,
            st.session_state.scaler_white,
            white_wine,
            "White"
        )
    else:
        st.info("ℹ️ Train the White wine model first on the 'Train & Tune' page.")

    st.markdown("---")

    # -----------------
    # Live Prediction
    # -----------------
    st.subheader("🔮 Live Quality Prediction")
    wine_choice = st.selectbox("Choose Wine Type", ["Red", "White"])
    
    model_key = f"best_model_{wine_choice.lower()}"
    scaler_key = f"scaler_{wine_choice.lower()}"
    
    # Check if model is trained
    if model_key not in st.session_state or scaler_key not in st.session_state:
        st.error(f"⚠️ Please train the {wine_choice} wine model first on the 'Train & Tune' page.")
    else:
        dataset = red_wine if wine_choice == "Red" else white_wine
        
        st.write("**Adjust the feature values below:**")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        feature_values = []
        
        for i, feat in enumerate(features):
            col = col1 if i % 2 == 0 else col2
            with col:
                val = st.slider(
                    f"{feat}",
                    min_value=float(dataset[feat].min()),
                    max_value=float(dataset[feat].max()),
                    value=float(dataset[feat].mean()),
                    step=0.01,
                    key=f"slider_{feat}"
                )
                feature_values.append(val)

        if st.button("🎯 Predict Quality", type="primary"):
            input_arr = np.array(feature_values).reshape(1, -1)
            scaler = st.session_state[scaler_key]
            input_scaled = scaler.transform(input_arr)
            model = st.session_state[model_key]

            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            
            # Display prediction
            st.success(f"🍷 Predicted {wine_choice} Wine Quality: **{pred}**")
            
            # Show probability distribution
            st.subheader("Prediction Confidence")
            prob_df = pd.DataFrame({
                'Quality': model.classes_,
                'Probability': proba * 100
            }).sort_values('Probability', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Quality', y='Probability', data=prob_df, palette='viridis', ax=ax)
            ax.set_ylabel("Probability (%)")
            ax.set_title("Quality Prediction Distribution")
            for i, v in enumerate(prob_df['Probability']):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom')
            st.pyplot(fig)

# =====================
# 4️⃣ Model Comparison Page
# =====================
elif page == "🔍 Model Comparison":
    st.header("Model Comparison")
    
    # Check if models are trained
    red_trained = "metrics_red" in st.session_state
    white_trained = "metrics_white" in st.session_state
    
    if not red_trained and not white_trained:
        st.warning("⚠️ No models have been trained yet. Please train models on the 'Train & Tune' page first.")
    else:
        # Comparison metrics
        if red_trained and white_trained:
            st.subheader("📊 Red vs White Wine Models")
            
            red_metrics = st.session_state.metrics_red
            white_metrics = st.session_state.metrics_white
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Red Wine': [
                    red_metrics['model_name'],
                    f"{red_metrics['accuracy']:.4f}",
                    f"{red_metrics['f1']:.4f}",
                    f"{red_metrics['precision']:.4f}",
                    f"{red_metrics['recall']:.4f}"
                ],
                'White Wine': [
                    white_metrics['model_name'],
                    f"{white_metrics['accuracy']:.4f}",
                    f"{white_metrics['f1']:.4f}",
                    f"{white_metrics['precision']:.4f}",
                    f"{white_metrics['recall']:.4f}"
                ]
            }, index=['Model Type', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
            
            st.dataframe(comparison_df)
            
            # Visual comparison
            metrics_data = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                'Red Wine': [red_metrics['accuracy'], red_metrics['f1'], 
                            red_metrics['precision'], red_metrics['recall']],
                'White Wine': [white_metrics['accuracy'], white_metrics['f1'],
                              white_metrics['precision'], white_metrics['recall']]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(metrics_data['Metric']))
            width = 0.35
            
            ax.bar(x - width/2, metrics_data['Red Wine'], width, label='Red Wine', color='#8B0000', alpha=0.8)
            ax.bar(x + width/2, metrics_data['White Wine'], width, label='White Wine', color='#4169E1', alpha=0.8)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_data['Metric'])
            ax.legend()
            ax.set_ylim([0, 1])
            
            # Add value labels on bars
            for i, (red, white) in enumerate(zip(metrics_data['Red Wine'], metrics_data['White Wine'])):
                ax.text(i - width/2, red + 0.02, f'{red:.3f}', ha='center', va='bottom', fontsize=9)
                ax.text(i + width/2, white + 0.02, f'{white:.3f}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
            
        elif red_trained:
            st.subheader("🍷 Red Wine Model Metrics")
            metrics = st.session_state.metrics_red
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("F1 Score", f"{metrics['f1']:.4f}")
            col3.metric("Precision", f"{metrics['precision']:.4f}")
            col4.metric("Recall", f"{metrics['recall']:.4f}")
            st.info("ℹ️ Train the White wine model to see comparison.")
            
        elif white_trained:
            st.subheader("🍷 White Wine Model Metrics")
            metrics = st.session_state.metrics_white
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("F1 Score", f"{metrics['f1']:.4f}")
            col3.metric("Precision", f"{metrics['precision']:.4f}")
            col4.metric("Recall", f"{metrics['recall']:.4f}")
            st.info("ℹ️ Train the Red wine model to see comparison.")







