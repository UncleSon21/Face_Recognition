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

def train_model(X_train, Y_train):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X_train, Y_train)
    print("Model training completed!")
    return model

def test_model(model, X_test_np, Y_test_np, X_test):
    # Initialize lists to hold the results
    predictions = []
    correct_labels = []

    # Loop through each of the first 500 images
    for i in range(500):
        test_image = X_test_np[i]

        # Predict the label of the test image
        label, confidence = model.predict(test_image)

        # Print the prediction and actual label
        print(f'Predicted Label: {labels[label]}, Confidence: {confidence}, Actual Label: {labels[Y_test_np[i]]}')

        # Check if the prediction is correct
        if labels[label] == labels[Y_test_np[i]]:
            img = X_test[i]  # The original image in color or grayscale

            # Annotate the image with the predicted label
            cv2.putText(img, str(labels[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
            cv2.rectangle(img, (0, 0), (250, 250), (0, 255, 0), 2)
            cv2.imshow('Detected Face', img)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Append the predicted label and the true label to the lists
        predictions.append(label)
        correct_labels.append(Y_test_np[i])

    # Calculate accuracy
    correct_predictions = sum([1 for p, c in zip(predictions, correct_labels) if p == c])
    accuracy = correct_predictions / len(correct_labels) * 100

    # Print results
    print(f"Total test images: {len(correct_labels)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    createLabels()
    importImages_Labels()

    print("Spliting train and test set")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    # Convert X and Y set to numpy array
    print("Converting X and Y set to numpy arrays")
    X_np = np.array(X)
    Y_np = np.array(Y)

    X_train_np = np.array(X_train)
    Y_train_np = np.array(Y_train)

    X_test_np = np.array(X_test)
    Y_test_np = np.array(Y_test)
    # Save X and Y set as .npy files

    print("Saving X and Y set as .npy files")
    X_np = np.save("X.npy", X_np)
    Y_np = np.save("Y.npy", Y_np)

    test_model(train_model(X_train_np, Y_train_np), X_test_np, Y_test_np, X_test)

    print("Data preprocessing complete!")