# Dry Bean Classification – ML Assignment 2

## 📌 Problem Statement
The goal of this assignment is to classify different types of dry beans using machine learning models.  
We train and evaluate six models on the Dry Bean dataset and compare their performance using standard metrics.  
Finally, we deploy a Streamlit app that allows interactive evaluation on a balanced test dataset.

---

## 📊 Dataset Description
- **Dataset:** Dry Bean Dataset 
- **Dataset Source:** Kaggle  
- **Instances in Balanced Test Set:** 595 (85 samples × 7 classes)  
- **Features:** 17 numerical attributes (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivalentDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1–4, etc.)  
- **Target Classes:** 7 bean types (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, `SIRA`)  

### Class Distribution (Balanced Test Set)
| Class     | Count |
|-----------|-------|
| BARBUNYA  | 85    |
| BOMBAY    | 85    |
| CALI      | 85    |
| DERMASON  | 85    |
| HOROZ     | 85    |
| SEKER     | 85    |
| SIRA      | 85    |

---

## 🤖 Models Trained
- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes (GaussianNB)  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

---

## 📈 Evaluation Metrics

| Model                  | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|-------------------------|----------|-------|-----------|--------|----------|-------|
| Logistic Regression     | 0.9277   | 0.9957 | 0.9287   | 0.9277 | 0.9280   | 0.9158 |
| Decision Tree           | 0.9782   | 0.9873 | 0.9781   | 0.9782 | 0.9781   | 0.9745 |
| KNN                     | 0.9345   | 0.9959 | 0.9351   | 0.9345 | 0.9346   | 0.9236 |
| Naive Bayes (GaussianNB)| 0.8975   | 0.9925 | 0.8982   | 0.8975 | 0.8975   | 0.8805 |
| Random Forest (Ensemble)| 0.9866   | 0.9975 | 0.9867   | 0.9866 | 0.9865   | 0.9843 |
| XGBoost (Ensemble)      | 0.9866   | 0.9995 | 0.9867   | 0.9866 | 0.9865   | 0.9843 |

---

## 🔎 Observations
| Model                  | Observation |
|-------------------------|-------------|
| Logistic Regression     | Strong baseline with high accuracy (92.7%) and excellent AUC (0.9957). Performs well on linear boundaries but slightly less robust compared to ensemble methods. |
| Decision Tree           | Very high accuracy (97.8%) and balanced metrics. Captures non-linear relationships effectively, but single trees can overfit compared to ensembles. |
| KNN                     | Solid accuracy (93.4%) and strong AUC (0.9959). Sensitive to scaling and dataset size; performance depends on choice of k. |
| Naive Bayes (GaussianNB)| Fast and simple with decent accuracy (89.7%) and AUC (0.9925). Assumes feature independence, which limits performance compared to more complex models. |
| Random Forest (Ensemble)| Excellent accuracy (98.6%) and metrics across the board. Robust ensemble method that reduces overfitting and generalizes well. |
| XGBoost (Ensemble)      | Matches Random Forest with top accuracy (98.6%) but achieves the highest AUC (0.9995). Computationally heavier, but delivers state-of-the-art performance. |

---

## 📂 Repository Structure


|-- 2025aa05803-ML2/

    │-- app.py
    │-- requirements.txt
    │-- README.md
    │-- 2025aa05803_ML_Assignment2.ipynb
    │-- test_data.csv
    │-- model/                     # Saved models + metrics
    │   │-- logistic_regression.pkl
    │   │-- decision_tree.pkl
    │   │-- knn.pkl
    │   │-- naive_bayes.pkl
    │   │-- random_forest.pkl
    │   │-- xgboost.pkl
    │   │-- logistic_regression_metrics.json
    │   │-- decision_tree_metrics.json
    │   │-- knn_metrics.json
    │   │-- naive_bayes_metrics.json
    │   │-- random_forest_metrics.json
    │   │-- xgboost_metrics.json
    
## 🚀 Streamlit App Features
- Upload test dataset (CSV)  
- Select model from dropdown  
- Display:  
  - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)  
  - Confusion matrix  
  - Predictions  

---

