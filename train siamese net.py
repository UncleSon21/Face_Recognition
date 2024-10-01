import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from model import siamese_model
from data_preprocessing import preprocess_data  # Import the preprocess_data function

# Helper function to generate pairs of images (positive and negative pairs) for training
def make_pairs(images, labels):
    pair_images = []
    pair_labels = []

    # Create a dictionary to map each label to its indices
    label_dict = {}
    for i, label in enumerate(labels):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(i)

    # Loop through each image and create pairs
    for i in range(len(images)):
        current_image = images[i]
        label = labels[i]

        # Positive pair (same label)
        pos_index = np.random.choice(label_dict[label])
        pos_image = images[pos_index]

        # Negative pair (different label)
        neg_label = np.random.choice([l for l in label_dict.keys() if l != label])
        neg_index = np.random.choice(label_dict[neg_label])
        neg_image = images[neg_index]

        # Append positive and negative pairs
        pair_images.append([current_image, pos_image])
        pair_labels.append([1])  # Positive pair label: 1

        pair_images.append([current_image, neg_image])
        pair_labels.append([0])  # Negative pair label: 0

    return np.array(pair_images), np.array(pair_labels)

# Contrastive loss function
def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# Compile the Siamese model with optimizer, loss, and evaluation metrics
def compile_model():
    siamese_net = siamese_model()
    siamese_net.compile(optimizer=Adam(0.0001), loss=contrastive_loss, metrics=['accuracy'])
    return siamese_net

# Visualize training progress
def visualise_training(history):
    ''' Plot the loss over epochs '''
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    ''' Plot the accuracy over epochs '''
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Train the Siamese network
def train_model(siamese_net, X_train_np, Y_train_np, X_test_np, Y_test_np):
    ''' Generate pairs for training and testing purposes '''
    train_pairs, train_labels = make_pairs(X_train_np, Y_train_np)
    test_pairs, test_labels = make_pairs(X_test_np, Y_test_np)

    ''' Reshape the input to fit the model's expected input shape (100, 100, 3) '''
    train_pair_1 = np.array(train_pairs[:, 0])
    train_pair_2 = np.array(train_pairs[:, 1])

    test_pair_1 = np.array(test_pairs[:, 0])
    test_pair_2 = np.array(test_pairs[:, 1])

    ''' Train model through the fit method '''
    history = siamese_net.fit([train_pair_1, train_pair_2], train_labels,  # input pairs
                              validation_data=([test_pair_1, test_pair_2], test_labels),  # validation pairs
                              batch_size=32,
                              epochs=10)

    ''' Save the model '''
    siamese_net.save('siamese_model.h5')

    ''' Visualize training progress '''
    visualise_training(history)

# Evaluate model based on the test set
def evaluate_model(siamese_net, X_test_np, Y_test_np):
    ''' Generate pairs for evaluation '''
    test_pairs, test_labels = make_pairs(X_test_np, Y_test_np)

    test_pair_1 = np.array(test_pairs[:, 0])
    test_pair_2 = np.array(test_pairs6[:, 1])

    ''' Evaluate the model '''
    test_loss, test_accuracy = siamese_net.evaluate([test_pair_1, test_pair_2], test_labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Main execution
if __name__ == '__main__':
    # Step 1: Preprocess the dataset
    X_train_np, Y_train_np, X_test_np, Y_test_np = preprocess_data()

    # Step 2: Compile and train the model
    siamese_net = compile_model()
    train_model(siamese_net, X_train_np, Y_train_np, X_test_np, Y_test_np)

    # Step 3: Evaluate the model
    evaluate_model(siamese_net, X_test_np, Y_test_np)
