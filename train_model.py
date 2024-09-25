import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import os

# import X and Y set
X = np.load('X.npy', allow_pickle=True)
Y = np.load('y.npy')

# Create labels set
labels = []

img_folder = 'archive/lfw-deepfunneled/lfw-deepfunneled'

# Method to restrain cpu's growth
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

def train_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X_train, Y_train)
    print("Model training completed!")
    return model

def test_model(model):
    # Initialize lists to hold the results
    predictions = []
    correct_labels = []

    # Loop through each first 200 images
    for i in enumerate(X_test[:500]):
        # Predict the label of the test image
        label, confidence = model.predict(i)

        print(f'Label: {labels[label]}, Confidence: {confidence}, Actual: {labels[Y_test[i]]}')

        if(labels[label] == labels[Y_test[i]]):
            cv2.putText(i, str(labels[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
            cv2.rectangle(i, (0, 0), (250, 250), (0, 255, 0), 2)
            cv2.imshow('Detected Face', i)


        # Append the predicted label and the true label to the lists
        predictions.append(label)
        correct_labels.append(Y_test[i])

        # Optionally, visualize the results (if needed)
        # You can load the image back if you want to display it
        # test_image = cv2.imread(test_image_path)  # Load the corresponding image
        # cv2.imshow(f"Prediction: {label}, Confidence: {confidence}", test_image)
        # cv2.waitKey(0)

    # Calculate accuracy
    correct_predictions = sum([1 for p, c in zip(predictions, correct_labels) if p == c])
    accuracy = correct_predictions / len(correct_labels) * 100

    # Print results
    print(f"Total test images: {len(correct_labels)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    restrained_cpu()
    createLabels()
    test_model(train_model())





