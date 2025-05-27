"""
This lab will guide you through creating, training, and evaluating models that use transpose convolution layers for tasks such as image reconstruction.

##### Learning objectives:
By the end of this lab, you will:
- Apply transpose convolution in practical scenarios using Keras.  
- Create, compile, train, and evaluate the model
- Visualize the results  

#### Steps:
**Step 1: Import Necessary Libraries**
"""
import warnings
warnings.simplefilter('ignore')
!pip install tensorflow==2.16.2
!pip install matplotlib
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D

import numpy as np
import matplotlib.pyplot as plt

"""
**Step 2: Define the Input Layer**
You need to define the input shape for your model. For simplicity, let's assume you are working with an input image of size 28x28 with 1 channel (grayscale).
"""
input_layer = Input(shape=(28, 28, 1))

"""
**Step 3: Add convolutional and transpose convolutional layers**
"""
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(conv_layer)

"""
**Step 4: Create the Model**
"""
model = Model(inputs=input_layer, outputs=transpose_conv_layer)

"""
**Step 5: Compile the Model**
"""
model.compile(optimizer='adam', loss='mean_squared_error')


"""
**Step 6: Train the Model**
You can now train the model on some training data. For this example, let's assume X_train is our training input data.
"""

# Generate synthetic training data
X_train = np.random.rand(1000, 28, 28, 1)
y_train = X_train # For reconstruction, the target is the input
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

"""
**Step 7: Evaluate the Model**
"""

# Generate synthetic test data
X_test = np.random.rand(200, 28, 28, 1)
y_test = X_test
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

"""
**Step 8: Visualize the Results**
"""
# Predict on test data
y_pred = model.predict(X_test)
# Plot some sample images

n = 10 # Number of samples to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()

"""
#### Exercise 1: Experiment with Different Kernel Sizes
**Objective:** Understand the impact of different kernel sizes on the model's performance.
**Instructions:**
1. Modify the kernel size of the `Conv2D` and `Conv2DTranspose` layers.
2. Recompile, train, and evaluate the model.
3. Observe and record the differences in performance.
"""
conv_layer_new = Conv2D(filters=32, kernel_size=(2, 2), activation='relu', padding='same')(input_layer)
transpose_conv_layer_new = Conv2DTranspose(filters=1, kernel_size=(2, 2), activation='sigmoid', padding='same')(conv_layer)
model = Model(inputs=input_layer, outputs=transpose_conv_layer_new)
model.compile(optimizer='adam', loss='mean_squared_error')
new_history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

#enlarging the kernel size for convolution: this increased the loss so led to worse performance
#opposite direction led to slightly worse results too

#enlarging kernel size transpose convolution: this led to slightly worse results
#opposite direction led to a bit better results!

#making both smaller led to the best result! More detail preserved, but this is computationally more expensive

"""#### Exercise 2: Add Dropout Layers
**Objective:** Add dropout layers to prevent overfitting.
**Instructions:**
1. Add dropout layers after the convolutional layer.
2. Set the dropout rate to 0.5.
3. Recompile, train, and evaluate the model.
"""
from tensorflow.keras.layers import Dropout
dropout_layer = Dropout(0.5)(conv_layer_new)
transpose_conv_layer_new = Conv2DTranspose(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(dropout_layer)
model = Model(inputs=input_layer, outputs = transpose_conv_layer_new)
model.compile(optimizer='adam', loss='mean_squared_error')
new_new_history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

#this led to a significant improvment in loss but a substantial decrease in validation loss, meaning the opposite of what we expected
#similarly for the test

"""#### Exercise 3: Use Different Activation Functions
**Objective:** Experiment with different activation functions and observe their impact on model performance.
**Instructions:**
1. Change the activation function of the convolutional and transpose convolutional layers to `tanh`.
2. Recompile, train, and evaluate the model.
"""
conv_layer_new = Conv2D(filters=32, kernel_size=(4, 4), activation='tanh', padding='same')(input_layer)
transpose_conv_layer_new = Conv2DTranspose(filters=1, kernel_size=(3,3), activation='tanh', padding='same')(dropout_layer)
model = Model(inputs=input_layer, outputs = transpose_conv_layer_new)
model.compile(optimizer='adam', loss='mean_squared_error')
new_new_history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

#this led to worse results compared to before

"""
### Conclusion:

By completing this lab, you have successfully created, trained, and evaluated a simple neural network model using transpose convolution for image reconstruction. This exercise provided hands-on experience with Keras and practical applications of transpose convolution layers. Continue experimenting with different architectures and datasets to deepen your understanding and skills in deep learning with Keras.

Copyright © IBM Corporation. All rights reserved.
"""

