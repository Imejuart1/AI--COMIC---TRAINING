import cv2
import os
import numpy as np

def load_dataset(data_dir):

    sketchy_images = []
    thicker_images = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg ') or file_name.endswith('.png'):
            #load sketchy image 
            sketchy_imagepath = os.path.join(data_dir, file_name)
            sketchy_img = cv2.imread(sketchy_imagepath, cv2.IMREAD_GRAYSCALE)
            #Load thicker iamge
            thicker_imagepath= os.path.join(data_dir, file_name)
            thicker_img = cv2.imread(thicker_imagepath, cv2.IMREAD_GRAYSCALE)

            if sketchy_img is not None and thicker_img is not None:
                sketchy_images.append(sketchy_img)
                thicker_images.append(thicker_img)

    # convet lists to numpy arrays.
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






        


