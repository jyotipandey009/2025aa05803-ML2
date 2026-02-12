import streamlit as st
import pandas as pd
import joblib
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Dry Bean Classification", layout="wide")
st.title("ML Assignment 2 – Dry Bean Classification Models")

st.markdown("""
Upload a test dataset, select a trained model, and view predictions, 
evaluation metrics, confusion matrix, and classification report.
""")

# -------------------------------
# Sidebar: Upload + Model Selection
# -------------------------------
st.sidebar.header("Controls")

# Dataset Upload in sidebar
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file with features (and optional 'Class' column)",
    type=["csv"]
)

# Model Selection in sidebar
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Provide Sample Dataset Download in sidebar
try:
    with open("test_data.csv", "r") as f:
        sample_csv = f.read()
    st.sidebar.download_button(
        label="Download test_data.csv",
        data=sample_csv,
        file_name="test_data.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.sidebar.warning("test_data.csv not found in repo. Please generate it first.")

# -------------------------------
# Map model names to filenames
# -------------------------------
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# -------------------------------
# Main App Logic
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    model_path = f"model/{model_map[model_choice]}"
    scaler_path = "model/scaler.pkl"
    encoder_path = "model/label_encoder.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)

        # Apply scaler if available
        X = data.drop(columns=["Class"], errors="ignore")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)

        predictions = model.predict(X)

        # Decode predictions back to class names if encoder exists
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            predictions = encoder.inverse_transform(predictions)

        st.subheader("Predictions")
        st.write(predictions)

        # If true labels are present in uploaded file
        if "Class" in data.columns:
            y_true = data["Class"].astype(str)
            y_pred = pd.Series(predictions).astype(str)

            # Evaluation Metrics
            st.header("Evaluation Metrics")
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Confusion Matrix
            st.header("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

            # Metrics Visualization (after confusion matrix)
            st.header("Metrics Visualization")
            metrics_to_plot = report_df.drop(index=["accuracy", "macro avg", "weighted avg"])
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_to_plot[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Score")
            plt.title("Precision, Recall, F1-score per Class")
            st.pyplot(fig)

        else:
            st.warning("No 'Class' column found in uploaded CSV. Metrics and confusion matrix require true labels.")
    else:
        st.error(f"Model file not found: {model_path}")