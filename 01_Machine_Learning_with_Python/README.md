# Course 1: Machine Learning with Python

This folder contains coursework and projects completed for the **[Machine Learning with Python](https://www.coursera.org/learn/machine-learning-with-python?specialization=ai-engineer)** course, part of the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera.

## 🧠 Course Description

This course introduces the foundations of machine learning using Python, covering both supervised and unsupervised learning techniques. Learners explore algorithms such as regression, classification, clustering, and recommender systems while applying them using real-world data and libraries like Scikit-learn.

By the end of this course, you will be able to:

- Understand and implement supervised learning models, including linear and logistic regression, decision trees, support vector machines (SVM), K-nearest neighbors (KNN), random forests, and XGBoost, applying them to real-world problems such as fraud detection, medical prescription, and customer churn.
- Apply unsupervised learning techniques including K-means, DBSCAN, HDBSCAN, PCA, t-SNE, and UMAP for tasks such as customer segmentation, gallery clustering, and high-dimensional data visualization.
- Evaluate model performance using metrics such as ROC-AUC, accuracy, confusion matrices, and clustering evaluation techniques including Voronoi diagrams and silhouette scores.
- Apply regularization techniques (Ridge and LASSO) for regression and feature selection, and build end-to-end machine learning pipelines with hyperparameter tuning using GridSearchCV.
- Build basic recommender systems and classification models for complex datasets, including Titanic survival prediction and rainfall forecasting, leveraging real-world data and competitive benchmarks.
-Use Python libraries like Pandas, Scikit-learn, Matplotlib, and Seaborn to process data, build, and evaluate machine learning models effectively and reproducibly.

---

## 📂 Contents: The coding projects I worked on (20 projects).

- `simple_linear_regression.py`: Using simple Linear Regression to predict co2 omission for a car (part 1). Best R-squared 0.68.
- `mulitple-linear-regression`: Predicting co2 omission for car using several features simulateneously (part 2). Best R-Squared 0.89. 
- `logistic_regression`: Predicting churn of customers in telecommunications company.<p>
   <img src="Images/feature_importance.png" alt="Churn_importances" width="170"/> 
- `multi_class_classification.py`: Multi-Class Classification: building the OvsAll, OvsO and mulinomial logistic regressions for obesitas level prediction (eight classes). Going from OvA to OvO increased accuracy test set from 76% to 92%.
- `decision_trees.py`: Building a decision tree for prescribing the correct medical drug. Final result 81.7% accuracy on test set for six medicine.<p>
   <img src="Images/decision_trees_new.png" alt="Distribution drug" width="170"/> 
- `regression_trees_taxi_tip.py`: Predicting the taxi tip. R-squared was relatively low due to the high amount of noise in the tip, which meant that decresing the max_depth of the tree improved OOS MSE by 10%. 
- `decision_tree_svm_ccfraud.py`: For the Kaggle Data set 'Credit Card Fraud Detection' with Decision Trees and SVM using python APIs, utilising hinge loss, obtained a SVM ROC-AUC score: 0.986.
- `knn_classification.py`: KNN for predicting service category telecommunications customers. Four categories, optimal neighbours is around 40 giving accuracy of 0.404 on test set. <p>
  <img src="Images/hyperparam.png" alt="Number of Neighbours" width="170"/> <img src="Images/variables.png" alt="Correlation Matrix" width="170"/> 
- `random__forests__xgboost.py`: Utilising Random Forest and XGBoost to predict housing prices in California. Evaluating both algorithms accuracy and speed. On test: MSE RF was 0.2556 and XGBoost 0.2226, training time/testing time 17.777 versus 0.292 and 0.373 versus 0.016.
- `k-means-customer-seg.py`: Applying K-means for customer segmentation on simulated data simulated and on a real dataset, evaluating results using various color figures. <p>
  <img src="Images/Education_Age_Income.png" alt="Size of dot is education level (bigger is higher)" width="170"/> 
- `comparing_dbscan_hdbscan.py`: Comparing DBSCAN to HDBSCAN on clustering art gallaries and musea in Canada (WGS84 date, and Web marcator (EPSG:3857)). <p>
   <img src="Images/DBSCAN.png" alt="DBSCAN" width="170"/> <img src="Images/HDBSCAN.png" alt="HDBSCAN" width="170"/> 
- `pca.py`: (a) projecting 2D data onto prinicpal axis via PCA and (b) exploring 4 dimensional reduction for iris flowers. One PC explained about 72% of variance.
- `t-sne_umap.py`:Comparing t-SNE and UMAP, also against PCA, on feature space dimensions (on a synthetic make_blobs dataset).
- `evaluating_classification_models.py`: Evaluating classification models, models for predicting tumor being benign or malignant in breast cancer data set, adding gaussian noise for measurement error. Out-of-sample accuracy was KNN 0.926 and 0.971, and f1-score 0.93 and 0.97 for KNN and SVM respecitively.
- `evaluating_random_forest.py`: Implementing a random forest to predict median housing price based on various attributes, evaluating its peformance and feature importance. OOS MSE was 0.2556.
- `evaluating_k_means_clustering.py`: Generating synthetic data, creating k-means models, evaluate their perforamnce, investigate evaluation metrics for interpreting results (intution for the subjective practice of evaluating clustering results), Voronoi diagram.
   <img src="Images/Hyperparam_Tune.png" alt="hyperparamter tuning scores" width="170"/> <img src="Images/hyperhyper.png" alt="Different Cluster Result Comparison" width="80"/> 
- `regularization_in_linearregression.py`: Implementing regularization techniques such as RIDGE, evaluating their impact when there are outliers, select features using LASSO, on synthetic data.
- `ml_pipelines_and_gridsearchcv.py`: Building and evaluating a ML pipeline using the Pipeline class, building a grid-search implementation, implementing and evaluating a classification using iris data set, extract feature importance.
   <img src="Images/confusion.png" alt="confusion matrix of best KNN model" width="170"/>
- `practice_project`: Predicting whether a passenger of titanic survived based on attributes, implemented using ML Pipeline class. Random forest and logistic regression are compared, Random forest scores weight average f1 score of 0.81 and accuracy of 0.82 while logistic regression 0.86 and 0.83, suggesting that the second provides a better fit.
- `finalproject_ausweather_.py`: Final project: for a KAGGLE data set predicting wether there will be rainfall using random forest and logistic regression. This time random forest was slightly better (accuracy 83% over 81%). Doing this by feature engineering, a classifier pipeline with grid-search, performance measures and evaluations, different classifiers (updating the pipeline), hyperparameter tuning. Scored 16/20 points (80/100%) by AI evaluation system. Interested in the most important signals?
   <img src="Images/weather_drivers.png" alt="drivers rain" width="170"/>




---

## 🔧 Tools and Libraries

- Python
- Jupyter Notebooks
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

## 📌 Certificate Series

This is the first course in the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer).


