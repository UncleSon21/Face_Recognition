import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import numpy as np
def evaluate_model(model, test_data):
    # initialise metrics
    recall = Recall()
    precision = Precision()

    # evaluate on the entire test set
    for test_input, test_val, y_true in test_data.as_numpy_iterator():
        y_hat = model.predict([test_input, test_val])
        recall.update_state(y_true, y_hat)
        precision.update_state(y_true, y_hat)

    print(f"Recall: {recall.result().numpy():.4f}")
    print(f"Precision:{precision.result().numpy():.4f}")

    test_input, test_val, y_true = next(test_data.as_numpy_iterator())
    y_hat = model.predict([test_input, test_val])

    y_hat_class = [1 if prediction > 0.5 else 0 for prediction in y_hat]

    print("Predicted classes:", y_hat_class)
    print("True classes:     ", y_true)

    # Visualisation
    n = min(len(test_input), 3)
    plt.figure(figsize=(15, 5*n))
    for i in range(n):
        plt.subplot(n, 2, 2*i+1)
        plt.imshow(test_input[i])
        plt.title(f"Input image {i+1}")
        plt.axis('off')

        plt.subplot(n, 2, 2*i+2)
        plt.imshow(test_val[i])
        plt.title(f"{'Positive' if y_true[i] == 1 else 'Negative'} Example {i+1}\nPrediction: {'Same' if y_hat_class[i] == 1 else 'Different'}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    return recall.result().numpy(), precision.result().numpy()
