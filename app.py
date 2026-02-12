import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Dry Bean Classification", layout="wide")
st.title("ML Assignment 2 – Dry Bean Classification Models")

st.markdown("""
Upload a small test dataset, select a trained model, and view predictions, 
evaluation metrics, confusion matrix, and classification report.
""")

# -------------------------------
# Provide Sample Dataset Download
# -------------------------------
st.header("Download Sample Test Dataset")
sample_csv = """Area,Perimeter,MajorAxisLength,MinorAxisLength,AspectRation,Eccentricity,ConvexArea,EquivDiameter,Extent,Solidity,Roundness,Compactness,ShapeFactor1,ShapeFactor2,ShapeFactor3,ShapeFactor4,label
1500,150,50,30,1.67,0.75,1520,43.6,0.85,0.98,0.84,0.92,0.78,0.85,0.91,0.88,KidneyBean
2000,180,60,35,1.71,0.77,2025,50.5,0.87,0.97,0.82,0.91,0.79,0.86,0.90,0.89,BlackBean
2500,200,70,40,1.75,0.79,2550,56.4,0.88,0.96,0.81,0.90,0.80,0.87,0.89,0.90,BombayBean
"""
st.download_button(
    label="Download sample test_data.csv",
    data=sample_csv,
    file_name="test_data.csv",
    mime="text/csv"
)

# -------------------------------
# Dataset Upload
# -------------------------------
st.header("Upload Test Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with features (and optional 'label' column)", type=["csv"])

# -------------------------------
# Model Selection
# -------------------------------
st.header("Select Model")
model_choice = st.selectbox(
    "Choose a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# Map model names to filenames
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# -------------------------------
# Run Prediction and Show Metrics
# -------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    model_path = f"model/{model_map[model_choice]}"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        X = data.drop(columns=["label"], errors="ignore")
        predictions = model.predict(X)

        st.subheader("Predictions")
        st.write(predictions)

        # If true labels are present in uploaded file
        if "label" in data.columns:
            y_true = data["label"]
            y_pred = predictions

            # Evaluation Metrics
            st.header("Evaluation Metrics")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix
            st.header("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
        else:
            st.warning("No 'label' column found in uploaded CSV. Metrics and confusion matrix require true labels.")
    else:
        st.error(f"Model file not found: {model_path}")