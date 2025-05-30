"""# Final Project: League of Legends Match Predictor

Use the [league_of_legends_data_large.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv) file to perform the tasks.

### Step 1: Data Loading and Preprocessing
"""
!pip install pandas
!pip install scikit-learn
!pip install torch
!pip install matplotlib

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import matplotlib
import torch

data = pd.read_csv("league_of_legends_data_large.csv")

data.head()
X = data.drop(["win"], axis = 1)
y = data["win"]
X.shape
y.shape
#y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train.shape

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)

"""### Step 2: Logistic Regression Model
"""
## Write your code here
import torch.nn as nn
import torch.optim as optim


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dimension):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dimension, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

"""### Step 3: Model Training
"""

def opimisation_program(epochs, optimizer):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch number {epoch+1} has loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)
        y_pred_test = model(X_test)
    label_train = y_pred_train > 0.5
    label_test = y_pred_test > 0.5
    print("Train accuracy: ", torch.mean(((label_train.view(-1) == y_train.view(-1))).float()).item())
    print("Test accuracy: ", torch.mean(((label_test.view(-1) == y_test.view(-1))).float()).item())
    return y_pred_train, y_pred_test, model

epochs = 1000
model = LogisticRegressionModel(X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr = 0.01)
_, _, _ = opimisation_program(epochs, optimizer)

"""### Step 4: Model Optimization and Evaluation
"""

epochs = 1000
model = LogisticRegressionModel(X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
_, _, _ = opimisation_program(epochs, optimizer)

"""### Step 5: Visualization and Interpretation
"""

!pip install seaborn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluation(y_prob, y_true):

    #predicting based on probabilities
    y_pred = (y_prob > 0.5).int()

    #confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    #classification report
    print(classification_report(y_true, y_pred))

    #ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


model = LogisticRegressionModel(X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.01)
y_train_pred, y_test_pred, _ = opimisation_program(epochs, optimizer) #probabilites
print("Train data, first model")
evaluation(y_train_pred, y_train)
print("Test data, first model")
evaluation(y_test_pred, y_test)

model = LogisticRegressionModel(X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
y_train_pred, y_test_pred, _ = opimisation_program(epochs, optimizer) #probabilites
print("Train data, second model")
evaluation(y_train_pred, y_train)
print("Test data, second model")
evaluation(y_test_pred, y_test)

"""
### Step 6: Model Saving and Loading
"""
## Write your code here
# Save the model
model = LogisticRegressionModel(X_train.shape[1])
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
_, _, model = opimisation_program(epochs, optimizer) #probabilites
torch.save(model.state_dict(), 'regression_model_classification.pth')

# Load the model
model = LogisticRegressionModel(X_train.shape[1])
model.load_state_dict(torch.load('regression_model_classification.pth'))

# Ensure the loaded model is in evaluation mode
model.eval()

# Evaluate the loaded model
with torch.no_grad():
    # Make predictions on the training and test data
    y_train_pred = model(X_train)
    y_test_pred = model(X_test)

    # Evaluate the predictions
    print("Train data, second model")
    evaluation(y_train_pred, y_train)

    print("Test data, second model")
    evaluation(y_test_pred, y_test)

"""### Step 7: Hyperparameter Tuning
"""

## Write your code here
lr_test = [0.01, 0.05, 0.1]
for lr in lr_test:
    print("Current learning rate is: ", lr)
    model = LogisticRegressionModel(X_train.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=lr)
    _, _, _ = opimisation_program(epochs=100, optimizer=optimizer) #my program already calculates the accuracy by default

"""### Step 8: Feature Importance
"""
import pandas as pd
import matplotlib.pyplot as plt

# Extract the weights of the linear layer
weights = model.linear.weight.data.numpy().flatten()

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    "Feature": list(X.columns),
    "Importance": weights
})
feature_importance_df = feature_importance_df.assign(Abs_Importance=lambda df: df["Importance"].abs()).sort_values("Abs_Importance", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color='skyblue')
plt.xlabel("Weight (Importance)")
plt.title("Feature Importances from Logistic Regression")
plt.axvline(x=0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()

"""
#### Conclusion:

Congratulations on completing the project! In this final project, you built a logistic regression model to predict the outcomes of League of Legends matches based on various in-game statistics. This comprehensive project involved several key steps, including data loading and preprocessing, model implementation, training, optimization, evaluation, visualization, model saving and loading, hyperparameter tuning, and feature importance analysis. This project provided hands-on experience with the complete workflow of developing a machine learning model for binary classification tasks using PyTorch.

© Copyright IBM Corporation. All rights reserved.
"""

