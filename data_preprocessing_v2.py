import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

img_folder = 'archive/lfw-deepfunneled/lfw-deepfunneled'

X = []
Y = []
labels = []

# Create haar cascade model
haar_cascade = cv2.CascadeClassifier('haar_face.xml')

def restrained_cpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def createLabels():
    for subdir in os.listdir(img_folder):
        subdir_path = os.path.join(img_folder, subdir)

        # Only process directories
        if os.path.isdir(subdir_path):
            image_files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]

            # Check if the folder containing more than 1 image
            if len(image_files) > 1:
                labels.append(subdir)

def importImages_Labels():
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

                # detect face in the image (increase image size 10% and minimal neighbors = 3)
                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

                for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y + h, x:x + w]
                    X.append(faces_roi)
                    Y.append(labels.index(label))

            else:
                print(f"Warning: Could not load image {image_path}")

    print("Importing images and labels completed!")

if __name__ == '__main__':
    restrained_cpu()
    print("Creating label ---------------------------")
    createLabels()

    print("Importing images and labels --------------------------")
    start_time = time.time()
    importImages_Labels()
    end_time = time.time()

    time = end_time - start_time
    print(f"Running time: {time}")
    print(f"Length of X: {len(X)}")

    print("Spliting train and test set")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Convert X and Y set to numpy array
    print("Converting X and Y set to numpy arrays")
    X_np = np.array(X, dtype='object')
    Y_np = np.array(Y)

    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)

    X_test_np = np.array(X_test)
    Y_test_np = np.array(Y_test)
    # Save X and Y set as .npy files

    # print("Saving X and Y set as .npy files")
    # X_np = np.save("X.npy", X_np)
    # Y_np = np.save("Y.npy", Y_np)