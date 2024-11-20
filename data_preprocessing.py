import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

img_folder = 'archive/lfw-deepfunneled/lfw-deepfunneled'

X = []
Y = []
labels = []

# Update the Haar cascade file path
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the cascade file is loaded correctly
if haar_cascade.empty():
    raise Exception("Error loading Haar cascade file. Check the file path.")

def restrained_cpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def createLabels():
    """Create labels from the subdirectories in the image folder."""
    for subdir in os.listdir(img_folder):
        subdir_path = os.path.join(img_folder, subdir)

        # Only process directories
        if os.path.isdir(subdir_path):
            labels.append(subdir)

def importImages_Labels():
    """Load images and their corresponding labels from the dataset."""
    for label in labels:
        path = os.path.join(img_folder, label)
        # Loop through each image in the sub-folder
        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)

            # Load the image using OpenCV
            image = cv2.imread(image_path)

            if image is not None:
                # Convert image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect face in the image (increase image size 10% and minimal neighbors = 3)
                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

                for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y + h, x:x + w]

                    # Resize the face region to 100x100 pixels
                    faces_roi_resized = cv2.resize(faces_roi, (100, 100))

                    # Add the resized face to the dataset
                    X.append(faces_roi_resized)
                    Y.append(labels.index(label))

            else:
                print(f"Warning: Could not load image {image_path}")

    print("Importing images and labels completed!")

def preprocess_data():
    """Run preprocessing steps and return the train/test splits."""
    restrained_cpu()
    createLabels()
    importImages_Labels()

    print("Splitting train and test set")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Convert X and Y set to numpy arrays
    print("Converting X and Y set to numpy arrays")
    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)
    X_test_np = np.array(X_test)
    Y_test_np = np.array(Y_test)

    # Ensure images have 3 channels (grayscale images will be duplicated across the color channels)
    X_train_np = np.stack([X_train_np] * 3, axis=-1)
    X_test_np = np.stack([X_test_np] * 3, axis=-1)

    return X_train_np, Y_train_np, X_test_np, Y_test_np
