import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POS_PATH = os.path.join(BASE_DIR, 'data', 'positive')
NEG_PATH = os.path.join(BASE_DIR, 'data', 'negative')
ANC_PATH = os.path.join(BASE_DIR, 'data', 'anchor')
LFW_TAR = os.path.join(BASE_DIR, 'lfw.tar')

# Image processing
IMG_SIZE = (250, 250)
CROP_SIZE = (250, 250)

# Data augmentation
AUG_COUNT = 9
BRIGHTNESS_DELTA = 0.02
CONTRAST_RANGE = (0.6, 1.0)
JPEG_QUALITY_RANGE = (90, 100)
SATURATION_RANGE = (0.9, 1.0)

# Webcam
WEBCAM_INDEX = 0
CROP_SIZE = (250, 250)

# Model parameters (for future use)
EMBEDDING_DIM = 128
DETECTION_THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5

# Train-test split
TRAIN_SPLIT = 0.7

# Prefetch buffer size
PREFETCH_BUFFER = 8