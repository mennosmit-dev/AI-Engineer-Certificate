# All Libraries required for this lab are listed below.

!pip install numpy==2.0.2
!pip install pandas==2.2.2
!pip install tensorflow_cpu==2.18.0

"""#### To use Keras, you will also need to install a backend framework – such as TensorFlow.
If you install TensorFlow 2.16 or above, it will install Keras by default.
We are using the CPU version of tensorflow since we are dealing with smaller datasets.
You may install the GPU version of tensorflow on your machine to accelarate the processing of larger datasets

#### Suppress the tensorflow warning messages
We use the following code to  suppress the warning messages due to use of CPU architechture for tensoflow.
You may want to **comment out** these lines if you are using the GPU architechture
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import keras

import warnings
warnings.simplefilter('ignore', FutureWarning)

"""## Download and Clean the Data Set
"""
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)
concrete_data.head()

#### Let's check how many data points we have
"""
concrete_data.shape

"""So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.
Let's check the dataset for any missing values.
"""
concrete_data.describe()
concrete_data.isnull().sum()

"""The data looks very clean and is ready to be used to build our model.
#### Split data into predictors and target
The target variable in this problem is the concrete sample strength. Therefore, our predictors will be all the other columns.
"""
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

"""
Let's do a quick sanity check of the predictors and the target dataframes.
"""
predictors.head()
target.head()

"""Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.
"""
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

"""Let's save the number of predictors to *n_cols* since we will need this number when building our network.
"""
n_cols = predictors_norm.shape[1] # number of predictors

##  Import Keras Packages
##### Let's import the rest of the packages from the Keras library that we will need to build our regression model.
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

"""## Build a Neural Network
Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.
"""
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

"""The above function create a model that has two hidden layers, each of 50 hidden units.
Let's call the function now to create our model.
"""
# build the model
model = regression_model()

"""Next, we will train and test the model at the same time using the *fit* method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.
"""
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

"""<strong>You can refer to this [link](https://keras.io/models/sequential/) to learn about other functions that you can use for prediction or evaluation.</strong>
Feel free to vary the following and note what impact each change has on the model's performance:
1. Increase or decreate number of neurons in hidden layers
2. Add more hidden layers
3. Increase number of epochs
Now using the same dateset,try to recreate regression model featuring five hidden layers, each with 50 nodes and ReLU activation functions, a single output layer, optimized using the Adam optimizer.
"""
# Write your code here
new_model = Sequential()
new_model.add(Input(shape=(n_cols,)))
new_model.add(Dense(50, activation='relu'))
new_model.add(Dense(50, activation='relu'))
new_model.add(Dense(50, activation='relu'))
new_model.add(Dense(50, activation='relu'))
new_model.add(Dense(50, activation='relu'))
new_model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

"""
Train and evaluate the model simultaneously using the fit() method by reserving 10% of the data for validation and training the model for 100 epochs
"""
model.fit(predictors_norm, target, validation_split=0.1, epochs=100, verbose=2)

"""
Based on the results, we notice that:
- Adding more hidden layers to the model increases its capacity to learn and represent complex relationships within the data. This allows the model to better identify, as a result, the model becomes more effective at fitting the training data and potentially improving its predictions.
- By reducing the proportion of data set aside for validation and using a larger portion for training, the model has access to more examples to learn from. This additional training data helps the model improve its understanding of the underlying trends, which can lead to better overall performance.
### Thank you for completing this lab!

This notebook was created by [Alex Aklson](https://www.linkedin.com/in/aklson/). I hope you found this lab interesting and educational. Feel free to contact me if you have any questions!

<!--
## Change Log

|  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
|---|---|---|---|
| 2024-11-20  | 3.0  | Aman  |  Updated the library versions to current |
| 2020-09-21  | 2.0  | Srishti  |  Migrated Lab to Markdown and added to course repo in GitLab |



<hr>

## <h3 align="center"> © IBM Corporation. All rights reserved. <h3/>

## <h3 align="center"> &#169; IBM Corporation. All rights reserved. <h3/>
"""
