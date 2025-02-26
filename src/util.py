import cv2
import numpy as np

def process_image(image_path):
    """Loads an image, converts it to grayscale, and applies Gaussian blur."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred

def detect_edges(image):
    """Detects edges using the Canny algorithm."""
    edges = cv2.Canny(image, 100, 200)
    return edges
