""">>>

What is done in the code:
Support Vector Machine (SVM) and Decision trees: Credit Card Fraud Detection using python APIs for a Kaggle Data set, utilising hinge loss, obtained a SVM ROC-AUC score: 0.986."""

!pip install pandas==2.2.3
!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3

# Commented out IPython magic to ensure Python compatibility.
# Import the libraries we need to use in this lab
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

"""## Load the dataset

Execute the cell below to load the dataset to the variable `raw_data`. The code will fetch the data set for the URL and load the same to the variable. A snapshot of the dataset will be generated as an output.

"""

# download the dataset
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
raw_data=pd.read_csv(url)
raw_data

"""
Each row in the dataset represents a credit card transaction. As shown above, each row has 31 variables. One variable (the last variable in the table above) is called Class and represents the target variable. Your objective will be to train a model that uses the other variables to predict the value of the Class variable. Let's first retrieve basic statistics about the target variable.

Note: For confidentiality reasons, the original names of most features are anonymized V1, V2 .. V28. The values of these features are the result of a PCA transformation and are numerical. The feature 'Class' is the target variable and it takes two values: 1 in case of fraud and 0 otherwise. For more information about the dataset please visit this webpage: https://www.kaggle.com/mlg-ulb/creditcardfraud.
"""

# get the set of distinct classes
labels = raw_data.Class.unique()

# get the count of each class
sizes = raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

"""As shown above, the Class variable has two values: 0 (the credit card transaction is legitimate) and 1 (the credit card transaction is fraudulent). Thus, you need to model a binary classification problem. Moreover, the dataset is highly unbalanced, the target variable classes are not represented equally. This case requires special attention when training or when evaluating the quality of a model. One way of handing this case at train time is to bias the model to pay more attention to the samples in the minority class. The models under the current study will be configured to take into account the class weights of the samples at train/fit time.

It is also prudent to understand which features affect the model in what way. We can visualize the effect of the different features on the model using the code below.
"""

correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))

"""This clearly shows that some features affect the output Class more than the others. For efficient modeling, we may use only the most correlated features.

You will now prepare the data for training. You will apply standard scaling to the input features and normalize them using $L_1$ norm for the training models to converge quickly. As seen in the data snapshot, there is a parameter called `Time` which we will not be considering for modeling. Hence, features 2 to 30 will be used as input features and feature 31, i.e. Class will be used as the target variable.
"""

# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

"""
Now that the dataset is ready for building the classification models, you need to first divide the pre-processed dataset into a subset to be used for training the model (the train set) and a subset to be used for evaluating the quality of the model (the test set).
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
Compute the sample weights to be used as input to the train routine so that it takes into account the class imbalance present in this dataset.
"""

w_train = compute_sample_weight('balanced', y_train)

"""Using these sample weights, we may train the Decision Tree classifier. We also make note of the time it takes for training this model to compare it against SVM, later in the lab.

"""

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)

"""
Unlike Decision Trees, we do not need to initiate a separate sample_weight for SVMs. We can simply pass a parameter in the scikit-learn function.
"""

# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)

"""
Run the following cell to compute the probabilities of the test samples belonging to the class of fraudulent transactions.
"""

y_pred_dt = dt.predict_proba(X_test)[:,1]

"""Using these probabilities, we can evaluate the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score as a metric of model performance.
The AUC-ROC score evaluates your model's ability to distinguish positive and negative classes considering all possible probability thresholds. The higher its value, the better the model is considered for separating the two classes of values.

"""

roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

"""
Run the following cell to compute the probabilities of the test samples belonging to the class of fraudulent transactions.
"""

y_pred_svm = svm.decision_function(X_test)

"""You may now evaluate the accuracy of SVM on the test set in terms of the ROC-AUC score.

"""

roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

"""## Practice Exercises
Based on what you have learnt in this lab, attempt the following questions.

Q1. Currently, we have used all 30 features of the dataset for training the models. Use the `corr()` function to find the top 6 features of the dataset to train the models on.
"""

# your code goes here
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[0:6]
correlation_values

"""<details><summary>Click here for solution</summary>

```python
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[:6]
correlation_values
```

<br>
The answer should be 'V3','V10','V12','V14','V16' and 'V17'.
</details>

Q2. Using only these 6 features, modify the input variable for training.
"""

data = raw_data[['V17','V14','V12', 'V10','V16', 'V3']]
data = StandardScaler().fit_transform(data)
#data = data.values
data= normalize(data, norm = 'l1')

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)

"""<details><summary>Click here for solution</summary>
Replace the statement defining the variable `X` with the following and run the cell again.
<br>

```python
X = data_matrix[:,[3,10,12,14,16,17]]
```

</details>

Q3. Execute the Decision Tree model for this modified input variable. How does the value of ROC-AUC metric change?
"""

w_train = compute_sample_weight('balanced', y_train)
dt = DecisionTreeClassifier(max_depth = 4, random_state = 35)
dt.fit(X_train, y_train, sample_weight= w_train)
y_pred_dt = dt.predict_proba(X_test)[:,1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

"""<details><summary>Click here for solution</summary>
You should observe an increase in the ROC-AUC value with this change for the Decision Tree model.
</details>

Q4. Execute the SVM model for this modified input variable. How does the value of ROC-AUC metric change?
"""

svm = LinearSVC(class_weight=  'balanced', random_state=31, loss="hinge", fit_intercept=False)
svm.fit(X_train, y_train)

predictions = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

"""<details><summary>Click here for solution</summary>
You should observe a decrease in the ROC-AUC value with this change for the SVM model.
</details>

Q5. What are the inferences you can draw about Decision Trees and SVMs with what you have learnt in this lab?

<details><summary>Click here for solution</summary>

- With a larger set of features, SVM performed relatively better in comparison to the Decision Trees.
- Decision Trees benefited from feature selection and performed better.
- SVMs may require higher feature dimensionality to create an efficient decision hyperplane.
</details>

### Congratulations! You're ready to move on to your next lesson!

## Author

<a href="https://www.linkedin.com/in/abhishek-gagneja-23051987/" target="_blank">Abishek Gagneja</a>


 ### Other Contributors

<a href="https://www.linkedin.com/in/jpgrossman/" target="_blank">Jeff Grossman</a>

<!--
## Changelog

| Date | Version | Changed by | Change Description |

|:------------|:------|:------------------|:---------------------------------------|

| 2024--1405 | 1.|0  Abhishek Gagnean    | Update content and practice exercis|es>

## <h3 align="center"> © IBM Corporation. All rights reserved. <h3/>
"""
