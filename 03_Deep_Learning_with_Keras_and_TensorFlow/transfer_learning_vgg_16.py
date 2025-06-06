"""
# Lab: Transfer Learning Implementation
In this lab, you will learn to implement transfer learning using a pre-trained model in Keras.

#### Learning objectives
 - Import necessary libraries and load the dataset.
 - Load a pre-trained model, VGG16, excluding the top layers.
 - Add new layers on top of the base model and compile the model.
 - Train the model on the new dataset.
 - Unfreeze some of the layers of the pre-trained model and fine-tune them.

#### Step 1: Setup the Environment
"""
!pip install tensorflow==2.16.2 matplotlib==3.9.1
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
#### Step 2: Load Pre-trained Model
"""
# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

"""#### Step 3: Create and Compile the Model
"""
# Create a new model and add the base model and new layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Change to the number of classes you have
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""### **Create Placeholder Images**
"""
import os
from PIL import Image
import numpy as np

# Create directories if they don't exist
os.makedirs('sample_data/class_a', exist_ok=True)
os.makedirs('sample_data/class_b', exist_ok=True)
# Create 10 sample images for each class
for i in range(10):
    # Create a blank white image for class_a
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
    img.save(f'sample_data/class_a/img_{i}.jpg')

    # Create a blank black image for class_b
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    img.save(f'sample_data/class_b/img_{i}.jpg')
print("Sample images created in 'sample_data/'")

"""#### Step 4: Train the Model
"""
# Load and preprocess the dataset
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Verify if the generator has loaded images correctly
print(f"Found {train_generator.samples} images belonging to {train_generator.num_classes} classes.")
# Train the model
if train_generator.samples > 0:
    model.fit(train_generator, epochs=10)

"""#### Step 5: Fine-Tune the Model
"""
# Unfreeze the top layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True
# Compile the model again
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model again
model.fit(train_generator, epochs=10)

"""### Exercises
#### Exercise 1: Visualize Training and Validation Loss
**Objective:** Plot the training and validation loss to observe the learning process of the model.
**Instructions:**
1. Modify the training code to include validation data.
2. Plot the training and validation loss for each epoch.
"""
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
plt.figure(figsize=(4,12))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('training and validaton loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""#### Exercise 2: Experiment with Different Optimizers
**Objective:** Experiment with different optimizers and observe their impact on model performance.
"""

print('sgd')
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=validation_generator)
print('rmsprop')
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=validation_generator)

"""
#### Exercise 3: Evaluate the Model on a Test Set
**Objective:** Evaluate the fine-tuned model on an unseen test set to assess its generalization performance.
**Instructions:**
1. Load a separate test set.
2. Evaluate the model on this test set and report the accuracy and loss.
"""
# This is actually not completely correct as you would need a seperate test data set!!!! is what I found out

# Load and preprocess the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Evaluate the fine-tuned model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

"""

### Summary

By completing these exercises, students will:
1. Visualize the training and validation loss to gain insights into the training process.
2. Experiment with different optimizers to understand their impact on model performance.
3. Evaluate the fine-tuned model on an unseen test set to assess its generalization capability.

#### Conclusion

Congratulations! In this lab, you have successfully implemented transfer learning using a pre-trained model in Keras. This lab exercise demonstrated how to train and fine-tune the model by unfreezing some of the layers.

Copyright © IBM Corporation. All rights reserved.
"""
