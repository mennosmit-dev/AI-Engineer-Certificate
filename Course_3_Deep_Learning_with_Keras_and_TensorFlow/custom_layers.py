
""" **Lab: Creating Custom Layers and Models**
By the end of this lab, you will:
- Create custom layers and integrate them into a Keras model
- Compile, train, and evaluate the model

**Step 1: Import libraries**
"""
!pip install tensorflow==2.16.2
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

"""**Step 2: Define a custom layer**
Define a custom dense layer with 32 units and ReLU activation.
"""
class CustomDenseLayer(Layer):
    def __init__(self, units=32):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

"""**Step 3: Integrate the custom layer into a model**
Create a Keras model using the custom layer.
"""
from tensorflow.keras.layers import Softmax
# Define the model with Softmax in the output layer
model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10),  # Hidden layer with ReLU activation
    Softmax()              # Output layer with Softmax activation for multi-class classification
])

"""The **Softmax** activation function is used in the output layer for multi-class classification tasks, ensuring the model outputs probabilities that sum up to 1 for each class, which aligns with categorical cross-entropy as the loss function. This adjustment ensures the model is optimized correctly for multi-class classification.
**Step 4: Compile the model**
Compile the model with the Adam optimizer and categorical cross-entropy loss.
"""
model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building:")
model.summary()
# Build the model to show parameters
model.build((1000, 20))
print("\nModel summary after building:")
model.summary()

"""**Step 5: Train the model**
Train the model on some example data. For this example, you will generate random data for training. In practice, use a real data set.
"""
import numpy as np
# Generate random data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(10, size=(1000, 1))
# Convert labels to categorical one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
model.fit(x_train, y_train, epochs=10, batch_size=32)

"""**Step 6: Evaluate the model**
Evaluate the model using test data to see its performance.
For this example, you will generate random test data. In practice, use a real data set.
"""
# Generate random test data
x_test = np.random.random((200, 20))
y_test = np.random.randint(10, size=(200, 1))
# Convert labels to categorical one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
# Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}')

"""### Exercises
#### Exercise 1: Visualize Model Architecture
**Objective:** Visualize the architecture of the custom Keras model to understand its structure.
**Instructions:**
1. Use the `plot_model` function from `tensorflow.keras.utils` to visualize the model architecture.
2. Save the plot as an image file.
"""
!pip install pydot==3.0.4
import pydot
from tensorflow.keras.utils import plot_model
#plot_model(model, to_file='model_architecture.png', show_tapes = True, show_layer_names = True)
plot_model(model, to_file='model_architecture.png', show_layer_names=True)

"""
#### Exercise 2: Add Dropout Layer
**Objective:** Enhance the model by adding a Dropout layer to prevent overfitting.
**Instructions:**
1. Add a Dropout layer between the custom dense layers.
2. Recompile the model and observe the impact on training.
"""
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
model2 = Sequential([
    Input(shape=(20,)),
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10)
])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs = 10, batch_size = 32)
loss, accuracy = model2.evaluate(x_test, y_test)

"""
#### Exercise 3: Adjust the Number of Units in Custom Layer
**Objective:** Experiment with different numbers of units in the custom dense layer to observe the impact on performance.
**Instructions:**
1. Change the number of units in the `CustomDenseLayer` to 128.
2. Recompile, train, and evaluate the model.
"""

model2 = Sequential([
    Input(shape=(20,)),
    CustomDenseLayer(128),
    CustomDenseLayer(10)
])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs = 10, batch_size = 32)

loss, accuracy = model2.evaluate(x_test, y_test)

"""
### Summary
By completing these exercises, students will:
1. Visualize the architecture of their custom Keras model.
2. Understand the impact of adding Dropout layers to prevent overfitting.
3. Experiment with different configurations of the custom dense layer to observe performance changes.
## Conclusion

Congratulations! You have successfully created and trained a custom layer in Keras. This lab exercise demonstrated how to extend Keras’s capabilities by creating custom layers and integrating them into a model.
"""

Copyright © IBM Corporation. All rights reserved.
