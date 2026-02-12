import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Dry Bean Classification - ML Assignment 2")

# --- Balanced Test Data Section ---
try:
    test_df = pd.read_csv("data/test_data.csv")   # adjust path if needed
    st.subheader("Balanced Test Data Preview")
    st.dataframe(test_df.head())

    # Download button for test dataset
    csv_test = test_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Balanced Test Dataset",
        data=csv_test,
        file_name="test_data.csv",
        mime="text/csv",
    )
except FileNotFoundError:
    st.warning("Balanced test_data.csv not found. Please add it to the repo.")

# --- Model Comparison Table Section ---
try:
    comparison_df = pd.read_csv("model/model_comparison.csv")
    st.subheader("Model Comparison Table")
    st.dataframe(comparison_df.style.format("{:.4f}"))

    # Download button for comparison table
    csv_comp = comparison_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Model Comparison Table",
        data=csv_comp,
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    # Plot Accuracy & F1 Score
    st.subheader("Accuracy & F1 Score Comparison")
    st.bar_chart(comparison_df.set_index("Model")[["Accuracy", "F1 Score"]])
except FileNotFoundError:
    st.warning("model_comparison.csv not found. Please add it to the repo.")

# --- Interactive Model Evaluation Section ---
uploaded_file = st.file_uploader("Upload balanced test_data.csv for evaluation", type="csv")
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Test data shape:", test_df.shape)
    st.write("Class distribution:", test_df["Class"].value_counts())

    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    model_choice = st.selectbox(
        "Select a model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    model_path = f"model/{model_choice.replace(' ', '_').lower()}.pkl"
    try:
        model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average="weighted"))
        st.write("Recall:", recall_score(y_test, y_pred, average="weighted"))
        st.write("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
        st.write("MCC:", matthews_corrcoef(y_test, y_pred))

        # AUC (multi-class support)
        try:
            y_proba = model.predict_proba(X_test)
            st.write("AUC:", roc_auc_score(y_test, y_proba, multi_class="ovr"))
        except Exception:
            st.write("AUC not available for this model")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}. Please add it to the repo.")