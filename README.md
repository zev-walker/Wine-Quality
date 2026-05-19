# 🍷 Wine Quality Prediction App

A machine learning web app that predicts the quality of red and white wine based on physicochemical properties, with model training, evaluation, live prediction, and side-by-side model comparison — all built with Streamlit.

---

## 📦 Modules

### 1. 📊 Data Exploration
- View sample rows from both red and white wine datasets
- Quality distribution charts for both wine types
- Dataset shape and missing value check

### 2. ⚙️ Train & Tune
- Choose between **Random Forest** or **SVM** classifier
- Choose wine type: **Red** or **White**
- Automatically runs **GridSearchCV** (3-fold CV) to find best hyperparameters
- Displays: Accuracy, F1 Score, Precision, Recall
- Shows full **Classification Report** and **Confusion Matrix**
- **Feature Importance** chart for Random Forest
- **Mean |Coefficient|** chart for SVM with linear kernel
- Download the trained model as a `.pkl` file

### 3. 📈 Evaluation & Prediction
- **5-Fold Cross-Validation** with mean F1 score and std deviation for trained models
- **Live Quality Prediction** — adjust wine feature sliders and get an instant quality prediction with a probability confidence chart

### 4. 🔍 Model Comparison
- Side-by-side comparison of Red vs White wine model metrics (Accuracy, F1, Precision, Recall)
- Bar chart visualization of both models' performance

---

## 🤖 Models & Hyperparameter Search

| Model | Parameters Tuned |
|---|---|
| Random Forest | `n_estimators`: [50, 100, 200], `max_depth`: [None, 10, 20] |
| SVM | `C`: [0.1, 1, 10], `kernel`: [linear, rbf] |

Both models are wrapped in a **Scikit-learn Pipeline** with `StandardScaler` to prevent data leakage.

---

## 📁 Dataset

- `winequality-red.csv` — Red wine physicochemical data
- `winequality-white.csv` — White wine physicochemical data
- Source: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Delimiter: semicolon (`;`)
- Target column: `quality`

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| ML Models | Scikit-learn (Random Forest, SVM) |
| Hyperparameter Tuning | GridSearchCV |
| Data Processing | Pandas, NumPy |
| Visualizations | Matplotlib, Seaborn |
| Model Export | Pickle |

---

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📝 Notes

- Train a model on the **Train & Tune** page before using **Evaluation & Prediction** or **Model Comparison**
- Models are stored in Streamlit session state — restarting the app clears them (use the download button to save)
- SVM feature coefficient chart only appears when kernel is `linear`
