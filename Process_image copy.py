import cv2
import os
import numpy as np


import cv2
import os
import numpy as np

def load_dataset(data_dir):
    """
    Load the dataset of paired sketchy line drawings and thicker line versions.
    """
    sketchy_images = []
    thicker_images = []

    # Iterate through the files in the data directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            # Load sketchy image
            sketchy_path = os.path.join(data_dir, file_name)
            sketchy_img = cv2.imread(sketchy_path, cv2.IMREAD_GRAYSCALE)
            # Load thicker image
            thicker_path = os.path.join(data_dir, 'thicker_' + file_name)  # Assuming thicker images have "thicker_" prefix
            thicker_img = cv2.imread(thicker_path, cv2.IMREAD_GRAYSCALE)

            # Check if images are loaded successfully
            if sketchy_img is not None and thicker_img is not None:
                sketchy_images.append(sketchy_img)
                thicker_images.append(thicker_img)

    # Convert lists to numpy arrays
    sketchy_images = np.array(sketchy_images)
    thicker_images = np.array(thicker_images)

    return sketchy_images, thicker_images

def preprocess_images(images):
    """
    Preprocess images (e.g., resize, normalize).
    """
    # Resize images to a common size (e.g., 256x256)
    resized_images = [cv2.resize(img, (256, 256)) for img in images]

    # Normalize pixel values to range [0, 1]
    normalized_images = np.array(resized_images) / 255.0

    return normalized_images

# Example usage
data_dir = 'path/to/your/dataset'
sketchy_images, thicker_images = load_dataset(data_dir)
sketchy_images = preprocess_images(sketchy_images)
thicker_images = preprocess_images(thicker_images)

