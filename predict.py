import tensorflow as tf
import config
from model import L1Dist
from config import MODEL_PATH, load_model
import numpy as np
from preprocess_images import preprocess
import os
import cv2
import matplotlib.pyplot as plt

def verify(model, detection_threshold, verification_threshold):
    results = []
    best_match_score = -1
    best_match_path = None
    input_path = os.path.join('application_data', 'input_image', 'input_image.jpg')

    if not os.path.exists(input_path):
        print("Input image not found. Please capture an image first.")
        return results, False, None

    input_img = preprocess(input_path)

    if input_img is None:
        print("Failed to preprocess input image")
        return results, False, None

    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        validation_img_path = os.path.join('application_data', 'verification_images', image)
        validation_img = preprocess(validation_img_path)

        if validation_img is None:
            print(f"Skipping comparison with {image} due to preprocessing error")
            continue

        result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result[0][0])

        if result[0][0] > best_match_score:
            best_match_score = result[0][0]
            best_match_path = validation_img_path

    if not results:
        print("No valid comparisons were made")
        return results, False, None

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold

    return results, verified, best_match_path

def real_time_verification():
    model = load_model()
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the full frame
        cv2.imshow('Verification', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            # Resize the frame to match the expected input size
            input_frame = cv2.resize(frame, config.IMG_SIZE)

            # Save the resized frame
            input_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
            cv2.imwrite(input_path, input_frame)

            results, verified, best_match_path = verify(model, 0.5, 0.5)
            print("Verified:", verified)
            print("Number of results above 0.9:", np.sum(np.squeeze(results) > 0.9))

            # Plot the results
            similarity_scores = np.squeeze(results)
            plot_verification_results(input_frame, best_match_path, similarity_scores, verified, threshold=0.5)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def plot_verification_results(input_image, best_match_path, similarity_scores, verified, threshold=0.5):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Display input image
    ax1.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Input Image")
    ax1.axis('off')

    # Display best matching image
    best_match_image = cv2.imread(best_match_path)
    best_match_image = cv2.resize(best_match_image, (input_image.shape[1], input_image.shape[0]))
    ax2.imshow(cv2.cvtColor(best_match_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Best Match\n{os.path.basename(best_match_path)}")
    ax2.axis('off')

    # Plot similarity scores
    bars = ax3.bar(range(len(similarity_scores)), similarity_scores, color='blue', alpha=0.5)
    ax3.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax3.set_xlabel("Comparison Index")
    ax3.set_ylabel("Similarity Score")
    ax3.set_title(f"Similarity Scores (Verified: {verified})")
    ax3.set_ylim(0, 1)  # Set y-axis limits for consistency

    # Color bars based on threshold
    for i, bar in enumerate(bars):
        if similarity_scores[i] > threshold:
            bar.set_color('green')

    ax3.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    real_time_verification()
