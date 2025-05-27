"""
# **Lab: Hyperparameter Tuning with Keras Tuner**

## Learning objectives:
By the end of this lab, you will:
- Install Keras Tuner and import the necessary libraries
- Load and preprocess the MNIST data set
- Define a model-building function that uses hyperparameters to configure the model architecture
- Set up Keras Tuner to search for the best hyperparameter configuration
- Retrieve the best hyperparameters from the search and build a model with these optimized values

### Exercise 1: Install the Keras Tuner
"""
!pip install tensorflow==2.16.2
!pip install keras-tuner==1.4.7
!pip install numpy<2.0.0
"""
import sys
# Increase recursion limit to prevent potential issues
sys.setrecursionlimit(100000)

"""#### Explanation:
The sys.setrecursionlimit function is used to increase the recursion limit, which helps prevent potential recursion errors when running complex models with deep nested functions or when using certain libraries like TensorFlow.
"""
# Step 2: Import necessary libraries
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import os
import warnings

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = ERROR only

# Step 3: Load and preprocess the MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')

"""
### Exercise 2: Defining the model with hyperparameters
"""
# Define a model-building function
def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

"""
### Exercise 3: Configuring the hyperparameter search
"""
# Create a RandomSearch Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Display a summary of the search space
tuner.search_space_summary()

"""
### Exercise 4: Running the hyperparameter search
"""
# Run the hyperparameter search
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# Display a summary of the results
tuner.results_summary()

"""
## Exercise 5: Analyzing and using the best hyperparameters
In this exercise, you retrieve the best hyperparameters found during the search and print their values. You then build a model with these optimized hyperparameters and train it on the full training data set. Finally, you evaluate the model’s performance on the test set to ensure that it performs well with the selected hyperparameters.
"""
# Step 1: Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The optimal number of units in the first dense layer is {best_hps.get('units')}.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Step 2: Build and Train the Model with Best Hyperparameters
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_val, y_val)
print(f'Test accuracy: {test_acc}')

"""
### Exercise 1: Setting Up Keras Tuner
#### Objective:
Learn how to set up Keras Tuner and prepare the environment for hyperparameter tuning.
"""
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

"""
### Exercise 2: Defining the model with hyperparameters
#### Objective:
Define a model-building function that uses hyperparameters to configure the model architecture.
"""
def function_builder(hp):
    model_new = Sequential([Flatten(input_shape=(28, 28)), Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'), Dense(10, activation='softmax')])
    model_new.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_new

"""
### Exercise 3: Configuring the hyperparameter search
#### Objective:
Set up Keras Tuner to search for the best hyperparameter configuration.
"""
search = kt.RandomSearch( function_builder, objective = 'val_accuracy', max_trials=10,
                         executions_per_trial=2, directory='my_dir', project_name='second_model')

"""
### Exercise 4: Running the hyperparameter search
#### Objective:
Run the hyperparameter search and dispaly the summary of the results.
"""
search.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

"""
### Exercise 5: Analyzing and using the best hyperparameters
#### Objective:
Retrieve the best hyperparameters from the search and build a model with these optimized values.
"""
best_params = search.get_best_hyperparameters(num_trials=1)[0]
print(f'Best Units: {best_params.get('units')} and best learning rate {best_params.get('learning_rate')}')
model_best = function_builder(best_params)
model.fit(x_train, y_train, epochs=10) #omitted validation split as we already seperated those earlier.

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_val, y_val)
print('Accuracy:')
print(test_acc)

"""
### Conclusion

Congratulations on completing this lab! You have learned to set up Keras Tuner and prepare the environment for hyperparameter tuning. In addition, you defined a model-building function that uses hyperparameters to configure the model architecture. You configured Keras Tuner to search for the best hyperparameter configuration and learned to run the hyperparameter search and analyze the results. Finally, you retrieved the best hyperparameters and built a model with these optimized values.

## Authors

Skillup

Copyright ©IBM Corporation. All rights reserved.
"""
