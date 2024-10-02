import os
import tensorflow as tf

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
POS_PATH = os.path.join(BASE_DIR, 'data', 'positive')
NEG_PATH = os.path.join(BASE_DIR, 'data', 'negative')
ANC_PATH = os.path.join(BASE_DIR, 'data', 'anchor')

# Dataset parameters
NUM_IMAGES_TO_LOAD = 3000
IMG_SIZE = (100, 100)
BATCH_SIZE = 16
BUFFER_SIZE = 10000

# Train-test split
TRAIN_SPLIT = 0.7

# Prefetch buffer size
PREFETCH_BUFFER = 8

# LFW dataset
LFW_TAR = os.path.join(BASE_DIR, 'lfw.tar')

# Webcam settings
WEBCAM_INDEX = 0

# Data augmentation settings
AUG_COUNT = 9
BRIGHTNESS_DELTA = 0.2
CONTRAST_RANGE = (0.8, 1.2)
JPEG_QUALITY_RANGE = (90, 100)
SATURATION_RANGE = (0.8, 1.2)

# GPU Configuration
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPUs found. Running on CPU.")

# Epochs
EPOCHS = 50
