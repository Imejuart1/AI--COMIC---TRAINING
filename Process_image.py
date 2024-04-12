import cv2
import os
import numpy as np

def load_dataset(data_dir):

    sketchy_images = []
    thicker_images = []

    for file_name in os.listdir(data_dir):
        

