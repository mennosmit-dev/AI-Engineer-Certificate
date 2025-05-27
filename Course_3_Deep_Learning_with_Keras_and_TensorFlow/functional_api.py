""" ## Lab: Implementing the Functional API in Keras
- Use the Keras Functional API to build a simple neural network model.
- Create an input layer, add hidden layers, and define an output layer using the Functional API.
"""

# Install TensorFlow
!pip install tensorflow==2.16.2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

"""**Explanation:**
**Step 2: Define the Input Layer**
You will define the input shape for your model. For simplicity, let's assume you are working with a dataset where each input is a vector of length 20.
"""
input_layer = Input(shape=(20,))
print(input_layer)

"""**Explanation:**
`Input(shape=(20,))` creates an input layer that expects input vectors of length 20.
`print(input_layer)` shows the layer information, helping you understand the type of information you can get about the layers.
**Step 3: Add Hidden Layers**
Next, you will add a couple of hidden layers to your model. Hidden layers help the model learn complex patterns in the data.
"""
hidden_layer1 = Dense(64, activation='relu')(input_layer)
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

"""**Explanation:**
`Dense(64, activation='relu')` creates a dense (fully connected) layer with 64 units and ReLU activation function.
Each hidden layer takes the output of the previous layer as its input.
**Step 4: Define the Output Layer**
Finally, you will define the output layer. Suppose you are working on a binary classification problem, so the output layer will have one unit with a sigmoid activation function.
"""
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

"""**Explanation:**
`Dense(1, activation='sigmoid')` creates a dense layer with 1 unit and a sigmoid activation function, suitable for binary classification.
**Step 5: Create the Model**
Now, you will create the model by specifying the input and output layers.
"""
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

"""**Explanation:**
`Model(inputs=input_layer, outputs=output_layer)` creates a Keras model that connects the input layer to the output layer through the hidden layers.
`model.summary()` provides a summary of the model, showing the layers, their shapes, and the number of parameters. This helps you interpret the model architecture.
**Step 6: Compile the Model**
Before training the model, you need to compile it. You will specify the loss function, optimizer, and evaluation metrics.
"""
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""**Explanation:**
`optimizer='adam'` specifies the Adam optimizer, a popular choice for training neural networks.
`loss='binary_crossentropy'` specifies the loss function for binary classification problems.
`metrics=['accuracy']` instructs Keras to evaluate the model using accuracy during training.
**Step 7: Train the Model**
You can now train the model using training data. For this example, let's assume `X_train` is your training input data and `y_train` is the corresponding label.
"""
# Example data (in practice, use real dataset)
import numpy as np
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))
model.fit(X_train, y_train, epochs=10, batch_size=32)

"""**Explanation:**
`X_train` and `y_train` are placeholders for your actual training data.
`model.fit` trains the model for a specified number of epochs and batch size.
**Step 8: Evaluate the Model**
After training, you can evaluate the model on test data to see how well it performs.
"""
# Example test data (in practice, use real dataset)
X_test = np.random.rand(200, 20)
y_test = np.random.randint(2, size=(200, 1))
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

"""**Explanation:**
`model.evaluate` computes the loss and accuracy of the model on test data.
`X_test` and `y_test` are placeholders for your actual test data.
### Dropout and Batch Normalization
Before we proceed with the practice exercise, let's briefly discuss two important techniques often used to improve the performance of neural networks: **Dropout Layers** and **Batch Normalization**.
#### Dropout Layers
Dropout is a regularization technique that helps prevent overfitting in neural networks. During training, Dropout randomly sets a fraction of input units to zero at each update cycle. This prevents the model from becoming overly reliant on any specific neurons, which encourages the network to learn more robust features that generalize better to unseen data.
**Key points:**
- Dropout is only applied during training, not during inference.
- The dropout rate is a hyperparameter that determines the fraction of neurons to drop.
#### Batch Normalization
Batch Normalization is a technique used to improve the training stability and speed of neural networks. It normalizes the output of a previous layer by re-centering and re-scaling the data, which helps in stabilizing the learning process. By reducing the internal covariate shift (the changes in the distribution of layer inputs), batch normalization allows the model to use higher learning rates, which often speeds up convergence.
**Key Points:**
- Batch normalization works by normalizing the inputs to each layer to have a mean of zero and a variance of one.
- It is applied during both training and inference, although its behavior varies slightly between the two phases.
- Batch normalization layers also introduce two learnable parameters that allow the model to scale and - shift the normalized output, which helps in restoring the model's representational power.
**Example of adding a Dropout layer in Keras:**
"""
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
# Define the input layer
input_layer = Input(shape=(20,))
# Add a hidden layer
hidden_layer = Dense(64, activation='relu')(input_layer)
# Add a Dropout layer
dropout_layer = Dropout(rate=0.5)(hidden_layer)
# Add another hidden layer after Dropout
hidden_layer2 = Dense(64, activation='relu')(dropout_layer)
# Define the output layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
# Summary of the model
model.summary()

"""**Example of adding Batch Normalization in Keras:**
"""
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.models import Model
# Define the input layer
input_layer = Input(shape=(20,))
# Add a hidden layer
hidden_layer = Dense(64, activation='relu')(input_layer)
# Add a BatchNormalization layer
batch_norm_layer = BatchNormalization()(hidden_layer)
# Add another hidden layer after BatchNormalization
hidden_layer2 = Dense(64, activation='relu')(batch_norm_layer)
# Define the output layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
# Summary of the model
model.summary()

"""### Practice exercises
#### Exercise 1: Add Dropout Layers
**Objective:** Learn to add dropout layers to prevent overfitting.
**Instructions:**
1. Add dropout layers after each hidden layer in the model.
2. Set the dropout rate to 0.5.
3. Recompile, train, and evaluate the model.
"""
input_layer = Input(shape=(20,))
hidden_layer1 = Dense(64, activation= 'relu')(input_layer)
dropout_1 = Dropout(0.5)(hidden_layer1)
hidden_layer2 = Dense(64, activation= 'relu')(dropout_1)
dropout_2 = Dropout(0.5)(hidden_layer2)
output_layer = Dense(1, activation='sigmoid')(dropout_2)
model = Model(inputs = input_layer, outputs = output_layer)
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size= 32)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

"""#### Exercise 2: Change Activation Functions
**Objective:** Experiment with different activation functions.
**Instructions:**
1. Change the activation function of the hidden layers from ReLU to Tanh.
2. Recompile, train, and evaluate the model to see the effect.
"""
input_layer = Input(shape=(20,))
hidden_layer1 = Dense(64, activation= 'tanh')(input_layer)
dropout_1 = Dropout(0.5)(hidden_layer1)
hidden_layer2 = Dense(64, activation= 'tanh')(dropout_1)
dropout_2 = Dropout(0.5)(hidden_layer2)
output_layer = Dense(1, activation='sigmoid')(dropout_2)
model = Model(inputs = input_layer, outputs = output_layer)
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size= 32)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

"""
#### Exercise 3: Use Batch Normalization
**Objective:** Implement batch normalization to improve training stability.
**Instructions:**
1. Add batch normalization layers after each hidden layer.
2. Recompile, train, and evaluate the model.
"""
input_layer = Input(shape=(20,))
hidden_layer1 = Dense(64, activation= 'relu')(input_layer)
dropout_1 = BatchNormalization()(hidden_layer1)
hidden_layer2 = Dense(64, activation= 'relu')(dropout_1)
dropout_2 = BatchNormalization()(hidden_layer2)
output_layer = Dense(1, activation='sigmoid')(dropout_2)
model = Model(inputs = input_layer, outputs = output_layer)
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size= 32)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

"""
### Summary
By completing these exercises, students will:
1. Understand the impact of dropout layers on model overfitting.
2. Learn how different activation functions affect model performance.
3. Gain experience with batch normalization to stabilize and accelerate training.
**Conclusion:**
You have successfully created, trained, and evaluated a simple neural network model using the Keras Functional API. This foundational knowledge will allow you to build more complex models and explore advanced functionalities in Keras.

Copyright © IBM Corporation. All rights reserved.
"""
