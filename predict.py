import tensorflow as tf

import config
from model import L1Dist
from config import MODEL_PATH, load_model
import numpy as np
from preprocess_images import preprocess
import os
import cv2
def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data','verification_images',image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection/len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified

def real_time_verification():
    model = load_model()
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the full frame without cropping
        cv2.imshow('Verification', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('v'):
            # Only crop when saving for verification
            input_frame = frame[120:120+250, 200:200+250, :]
            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), input_frame)
            results, verified = verify(model, 0.5, 0.5)
            print("Verified:", verified)
            print("Number of results above 0.9:", np.sum(np.squeeze(results) > 0.9))
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_verification()