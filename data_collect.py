import os
import cv2
import uuid
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import tensorflow as tf
import config

#GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
# Data augmentation
def data_aug(img):
    data = []
    for _ in range(config.AUG_COUNT):
        img = tf.image.stateless_random_brightness(img, max_delta=config.BRIGHTNESS_DELTA, seed=(1,2))
        img = tf.image.stateless_random_constrast(img, lower=config.CONTRAST_RANGE[0], upper=config.CONTRAST_RANGE[1], seed=(1, 3))
        img = tf.image.stateless_random_flip_left_right(img, seed(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=config.JPEG_QUALITY_RANGE[0], max_jpeg_quality=config.JPEG_QUALITY_RANGE[1], seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=config.SATURATION_RANGE[0], upper=config.SATURATION_RANGE[1], seed=(np.random.randint(100), np.random.randint(100)))

        data.append(img)
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
            # Crop and resize the frame for saving
            save_frame = cv2.resize(frame, (config.IMG_SIZE[1], config.IMG_SIZE[0]))
            imgname = os.path.join(config.ANC_PATH, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(imgname, save_frame)
            print(f"Anchor image saved: {imgname}")
        elif key == ord('p'):
            # Crop and resize the frame for saving
            save_frame = cv2.resize(frame, (config.IMG_SIZE[1], config.IMG_SIZE[0]))
            imgname = os.path.join(config.POS_PATH, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(imgname, save_frame)
            print(f"Positive image saved: {imgname}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

print("Starting webcam for collecting positive and anchor images...")
print("Press 'p' to save a positive image, 'a' to save an anchor image, and 'q' to quit.")
collect_data()

#Apply data augmentation
for path in [config.POS_PATH, config.ANC_PATH]:
    print(f"Applying data augmentation to images in {path}...")
    for file_name in os.listdir(path):
        img_path = os.path.join(path, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img)

        for image in augmented_images:
            aug_imgname = os.path.join(path, f'{uuid.uuid1()}.jpg')
            cv2.imwrite(aug_imgname, image.numpy())

print(f"Number of negative images: {len(os.listdir(config.NEG_PATH))}")
print(f"Number of positive images: {len(os.listdir(config.POS_PATH))}")
print(f"Number of anchor images: {len(os.listdir(config.ANC_PATH))}")

# Display a sample of preprocessed images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(cv2.cvtColor(cv2.imread(os.path.join(config.NEG_PATH, os.listdir(config.NEG_PATH)[0])), cv2.COLOR_BGR2RGB))
ax1.set_title("Negative Sample")
ax2.imshow(cv2.imread(os.path.join(config.POS_PATH, os.listdir(config.POS_PATH)[0])))
ax2.set_title("Positive Sample")
ax3.imshow(cv2.imread(os.path.join(config.ANC_PATH, os.listdir(config.ANC_PATH)[0])))
ax3.set_title("Anchor Sample")
plt.show()