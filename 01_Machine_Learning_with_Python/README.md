# Machine Learning with Python – Project Implementations

This folder contains foundational machine learning projects completed as part of the  
**IBM AI Engineer Professional Certificate**.  

The focus here is on building strong practical intuition for classical ML algorithms, 
model evaluation, feature engineering, and pipeline design — forming the statistical 
and modeling foundation behind my applied ML and reinforcement learning work.

---

## 🧠 Overview

Implemented a wide range of supervised and unsupervised learning techniques, including:

- Regression and classification models (Logistic Regression, SVM, KNN, Trees, Random Forest, XGBoost)
- Clustering methods (K-Means, DBSCAN, HDBSCAN)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Regularization (Ridge, Lasso)
- End-to-end pipelines with GridSearchCV
- Model evaluation using ROC-AUC, F1-score, confusion matrices, and clustering metrics

Libraries used: Scikit-learn, Pandas, NumPy, Matplotlib.

---

## 📂 Selected Implementations

### 🔹 Regression & Classification

- `simple_linear_regression.py` – CO₂ prediction using linear regression (R²: 0.68)  
- `multiple-linear-regression` – Multi-feature regression (R²: 0.89)  
- `logistic_regression` – Telecom churn prediction  
- `multi_class_classification.py` – Obesity classification (OvA → OvO improved accuracy 76% → 92%)  
- `decision_trees.py` – Drug prescription classifier (98.3% accuracy)  
- `decision_tree_svm_ccfraud.py` – Credit card fraud detection (SVM ROC-AUC: 0.986)

---

### 🔹 Tree-Based Models & Ensembles

- `random__forests__xgboost.py` – Housing price prediction (XGBoost MSE: 0.2226)  
- `regression_trees_taxi_tip.py` – Tree-based regression with noise analysis  
- `evaluating_random_forest.py` – Feature importance & model diagnostics  

---

### 🔹 Unsupervised Learning & Dimensionality Reduction

- `k-means-customer-seg.py` – Customer segmentation  
- `comparing_dbscan_hdbscan.py` – Density-based clustering comparison  
- `pca.py` – Principal component analysis (72% variance explained)  
- `t-sne_umap.py` – Visualization of high-dimensional feature spaces  

---

### 🔹 Model Evaluation & Pipelines

- `evaluating_classification_models.py` – Breast cancer classification benchmarking  
- `ml_pipelines_and_gridsearchcv.py` – Pipeline design with hyperparameter tuning  
- `regularization_in_linearregression.py` – Ridge & Lasso feature selection  

---

### 🔹 Applied Projects

- `practice_project` – Titanic survival prediction (Logistic Regression outperformed RF)  
- `finalproject_ausweather_.py` – Rainfall prediction pipeline (RF accuracy: 83%)

---

## 🔧 Tools & Libraries

Python • Scikit-learn • Pandas • NumPy • Matplotlib • Jupyter

---

## 📌 Context

This module forms the classical ML foundation within the  
IBM AI Engineering Professional Certificate and complements my work in deep learning, reinforcement learning, and production ML systems.
