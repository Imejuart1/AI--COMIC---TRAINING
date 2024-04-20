import cv2
import numpy as np
import os

# Define the path to your images directory
directory = r'C:\Users\USER\Documents\AI- COMIC - TRAINING\IMAGES-AI'

# List all files in the directory
files = os.listdir(directory)

# Load and preprocess images
preprocessed_sketches = []
preprocessed_thick_lines = []

for file in files:
    if file.startswith('sketchy'):
        # Load and preprocess sketch image
        sketch_path = os.path.join(directory, file)
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
        sketch_resized = cv2.resize(sketch, (256, 256))
        sketch_normalized = sketch_resized / 255.0
        sketch_normalized = np.expand_dims(sketch_normalized, axis=-1)
        preprocessed_sketches.append(sketch_normalized)
    elif file.startswith('thicker'):
        # Load and preprocess thick line image
        thick_line_path = os.path.join(directory, file)
        thick_line = cv2.imread(thick_line_path, cv2.IMREAD_GRAYSCALE)
        thick_line_resized = cv2.resize(thick_line, (256, 256))
        thick_line_normalized = thick_line_resized / 255.0
        thick_line_normalized = np.expand_dims(thick_line_normalized, axis=-1)
        preprocessed_thick_lines.append(thick_line_normalized)

# Convert lists to NumPy arrays
X_train = np.array(preprocessed_sketches)
y_train = np.array(preprocessed_thick_lines)

# Print the shape of the training data (for verification)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

import matplotlib.pyplot as plt

# Define a function to display images
def show_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.show()

# Display example images
show_images([X_train[0], y_train[0]], ['Sketch', 'Thick Lines'])

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model architecture
model = models.Sequential([
    # Encoder
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    
    # Decoder
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Load an example sketch image
sketchy_image_path = os.path.join(directory, 'sketchy.png')
sketchy_image = cv2.imread(sketchy_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if sketchy_image is None:
    print("Error: Failed to load the sketchy image.")
else:
    # Resize the sketchy image
    sketchy_resized = cv2.resize(sketchy_image, (256, 256))

    # Preprocess the resized sketchy image
    sketchy_normalized = sketchy_resized / 255.0
    sketchy_normalized = np.expand_dims(sketchy_normalized, axis=-1)

    # Expand dimensions to match the model's input shape
    sketchy_input = np.expand_dims(sketchy_normalized, axis=0)

    # Generate thicker lines for the sketchy image
    predicted_thick_lines = model.predict(sketchy_input)

    # Post-process the predicted thick lines if necessary
    # For example, you can rescale pixel values from [0, 1] to [0, 255]
    predicted_thick_lines_rescaled = predicted_thick_lines * 255.0

    # Display the original sketchy image and the predicted thick lines
    show_images([sketchy_normalized.squeeze(), predicted_thick_lines_rescaled.squeeze()], ['Sketchy', 'Thickened Lines'])


