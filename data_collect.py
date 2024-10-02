import os
import cv2
import uuid
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import tensorflow as tf
import config

config.configure_gpu()

# Create directories
for path in [config.POS_PATH, config.NEG_PATH, config.ANC_PATH]:
    os.makedirs(path, exist_ok=True)

# Extract LFW dataset
print("Extracting LFW dataset...")
try:
    with tarfile.open(config.LFW_TAR, "r") as tar:
        tar.extractall()
    print("Extraction completed successfully.")
except Exception as e:
    print(f"An error occurred during extraction: {e}")
    exit(1)

# Move LFW Images to negative folder
print("Moving LFW images to negative folder...")
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(config.NEG_PATH, f"{directory}_{file}")
        os.replace(EX_PATH, NEW_PATH)

# Remove the LFW folder
import shutil
shutil.rmtree('lfw')

def data_aug(img):
    if img is None:
        print("Error: Received None image in data_aug function")
        return []

    # Convert to float32 and scale to [0, 1]
    img = tf.cast(img, tf.float32) / 255.0

    data = []
    for _ in range(config.AUG_COUNT):
        try:
            # Apply augmentations
            aug_img = tf.image.stateless_random_brightness(img, max_delta=0.2,
                                                           seed=(np.random.randint(100), np.random.randint(100)))
            aug_img = tf.image.stateless_random_contrast(aug_img, lower=0.8, upper=1.2,
                                                         seed=(np.random.randint(100), np.random.randint(100)))
            aug_img = tf.image.stateless_random_saturation(aug_img, lower=0.8, upper=1.2,
                                                           seed=(np.random.randint(100), np.random.randint(100)))
            aug_img = tf.image.stateless_random_hue(aug_img, max_delta=0.1,
                                                    seed=(np.random.randint(100), np.random.randint(100)))

            # 50% chance of flipping the image horizontally
            if np.random.rand() > 0.5:
                aug_img = tf.image.flip_left_right(aug_img)

            # Ensure values are in [0, 1] range
            aug_img = tf.clip_by_value(aug_img, 0, 1)

            # Convert back to uint8 for saving
            aug_img = tf.cast(aug_img * 255.0, tf.uint8)

            data.append(aug_img)
        except Exception as e:
            print(f"Error during augmentation: {e}")
            continue

    return data


def collect_data():
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the frame to a larger size for display
        display_frame = cv2.resize(frame, (640, 480))

        # Show image to screen
        cv2.imshow('Image collection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            # Save anchor image
            save_and_augment(frame, config.ANC_PATH, "anchor")
        elif key == ord('p'):
            # Save positive image
            save_and_augment(frame, config.POS_PATH, "positive")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def save_and_augment(frame, path, img_type):
    # Save original image
    original_filename = os.path.join(path, f'{uuid.uuid1()}.jpg')
    cv2.imwrite(original_filename, frame)
    print(f"Original {img_type} image saved: {original_filename}")

    # Convert BGR to RGB for augmentation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform augmentation
    augmented_images = data_aug(frame_rgb)

    # Save augmented images
    for i, aug_img in enumerate(augmented_images):
        aug_filename = os.path.join(path, f'{uuid.uuid1()}_aug_{i}.jpg')
        cv2.imwrite(aug_filename, cv2.cvtColor(aug_img.numpy(), cv2.COLOR_RGB2BGR))
        print(f"Augmented {img_type} image saved: {aug_filename}")


print("Starting webcam for collecting positive and anchor images...")
print("Press 'p' to save a positive image, 'a' to save an anchor image, and 'q' to quit.")
collect_data()

# After collection, print the counts
print(f"Number of positive images: {len(os.listdir(config.POS_PATH))}")
print(f"Number of anchor images: {len(os.listdir(config.ANC_PATH))}")
