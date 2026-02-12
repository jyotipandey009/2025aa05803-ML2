# ML Assignment 2 – Classification Models (Dry Bean Dataset)

## Problem Statement
The objective of this assignment is to implement multiple machine learning classification models on the Dry Bean dataset, evaluate their performance using standard metrics, and deploy the models in an interactive Streamlit web application. This demonstrates an end-to-end ML workflow: modeling, evaluation, UI design, and deployment.

## Dataset Description
- **Dataset**: Dry Bean Dataset  
- **Source**: UCI Machine Learning Repository  
- **Features**: 16 shape-based features extracted from bean images  
- **Instances**: 13,611 samples across 7 classes of dry beans  
- **Task**: Multi-class classification to identify bean type based on morphological features  

## Models Used
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

## Final Comparison Table

| Model                | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
|----------------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression  | 0.921043 | 0.993389 | 0.921850  | 0.921043 | 0.921233 | 0.904609 |
| Decision Tree        | 0.892031 | 0.933376 | 0.891693  | 0.892031 | 0.891630 | 0.869569 |
| KNN                  | 0.917003 | 0.981302 | 0.917728  | 0.917003 | 0.917153 | 0.899648 |
| Naive Bayes          | 0.897907 | 0.990201 | 0.900702  | 0.897907 | 0.898075 | 0.877263 |
| Random Forest        | 0.920676 | 0.991447 | 0.920858  | 0.920676 | 0.920650 | 0.904052 |
| XGBoost              | 0.923981 | 0.993666 | 0.924149  | 0.923981 | 0.923936 | 0.908044 |

## Observations

| Model                | Observation |
|----------------------|-------------|
| Logistic Regression  | Achieved very high AUC (0.993), showing strong separability across classes. |
| Decision Tree        | Lowest performance among models, likely due to overfitting and weaker generalization. |
| KNN                  | Performed well with balanced metrics, but computationally expensive for large datasets. |
| Naive Bayes          | Delivered competitive AUC (0.990), but slightly lower MCC compared to Logistic Regression. |
| Random Forest        | Very strong performance, nearly matching Logistic Regression, with robust MCC (0.904). |
| XGBoost              | Achieved the best overall balance with highest accuracy (0.924) and MCC (0.908), confirming its strength as an ensemble method. |
