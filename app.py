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

# --- Download Balanced Test Dataset ---
try:
    test_df_sample = pd.read_csv("test_data.csv")
    st.subheader("Balanced Test Data Preview")
    st.dataframe(test_df_sample.head())

    csv_test = test_df_sample.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Balanced Test Dataset",
        data=csv_test,
        file_name="test_data.csv",
        mime="text/csv",
    )
except FileNotFoundError:
    st.warning("Balanced test_data.csv not found in repo.")

# --- Upload for Evaluation (Sidebar) ---
uploaded_file = st.sidebar.file_uploader("Upload balanced test_data.csv for evaluation", type="csv")
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Test data shape:", test_df.shape)
    st.write("Class distribution:", test_df["Class"].value_counts())

    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"].astype(str)  # ensure string type

    model_choice = st.sidebar.selectbox(
        "Select a model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    model_path = f"model/{model_choice.replace(' ', '_').lower()}.pkl"
    try:
        model = joblib.load(model_path)

        # Load label encoder
        try:
            label_encoder = joblib.load("model/label_encoder.pkl")
            y_pred = model.predict(X_test)
            # Decode predictions back to original class names
            y_pred_decoded = label_encoder.inverse_transform(y_pred)
            y_test_decoded = y_test
            class_names = label_encoder.classes_
        except Exception:
            # Fallback if encoder not available
            y_pred_decoded = model.predict(X_test).astype(str)
            y_test_decoded = y_test
            class_names = sorted(pd.unique(y_test_decoded))

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y_test_decoded, y_pred_decoded))
        st.write("Precision:", precision_score(y_test_decoded, y_pred_decoded, average="weighted", zero_division=0))
        st.write("Recall:", recall_score(y_test_decoded, y_pred_decoded, average="weighted", zero_division=0))
        st.write("F1 Score:", f1_score(y_test_decoded, y_pred_decoded, average="weighted", zero_division=0))
        st.write("MCC:", matthews_corrcoef(y_test_decoded, y_pred_decoded))

        # AUC requires numeric labels
        try:
            y_proba = model.predict_proba(X_test)
            y_test_enc = label_encoder.transform(y_test_decoded)
            st.write("AUC:", roc_auc_score(y_test_enc, y_proba, multi_class="ovr"))
        except Exception:
            st.write("AUC not available for this model")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}. Please add it to the repo.")
else:
    st.info("Please upload your balanced test_data.csv using the sidebar to proceed.")