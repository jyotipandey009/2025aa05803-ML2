import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# App Title and Introduction
# -------------------------------
st.set_page_config(page_title="Dry Bean Classification", layout="wide")
st.title("ML Assignment 2 – Dry Bean Classification Models")
st.markdown("""
This Streamlit app showcases the performance comparison of multiple ML models 
(Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost) 
on the **Dry Bean dataset**.
""")

# -------------------------------
# Load and Display Metrics
# -------------------------------
metrics_file = "model/model_metrics.csv"

st.header("Model Performance Comparison")

if os.path.exists(metrics_file):
    results_df = pd.read_csv(metrics_file)

    # Display table nicely
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "MCC": "{:.4f}"
    }), use_container_width=True)

    # Highlight best model
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    st.success(
        f"Best Model: **{best_model['Model']}** "
        f"(Accuracy: {best_model['Accuracy']:.4f}, MCC: {best_model['MCC']:.4f})"
    )
else:
    st.error("Metrics file not found. Please run the notebook to generate 'model/model_metrics.csv'.")

# -------------------------------
# Prediction Section
# -------------------------------
st.header("Try Predictions with a Saved Model")

uploaded_file = st.file_uploader("Upload a CSV file with features", type=["csv"])
model_choice = st.selectbox(
    "Select a model for prediction",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Map model names to filenames
    model_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    model_path = f"model/{model_map[model_choice]}"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        predictions = model.predict(data)
        st.subheader("Predictions")
        st.write(predictions)
    else:
        st.error(f"Model file not found: {model_path}")