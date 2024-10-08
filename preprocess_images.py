import tensorflow as tf
import numpy as np
import os
import config
import matplotlib.pyplot as plt


def is_image_file(filename):
    # List of valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def preprocess(file_path):
    if not is_image_file(os.path.basename(file_path)):
        print(f"Skipping non-image file: {file_path}")
        return None

    try:
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)

        # Detect the image format and decode accordingly
        if file_path.lower().endswith(('.png', '.gif', '.bmp')):
            img = tf.image.decode_image(byte_img, channels=3)
        else:
            # Default to JPEG for other formats
            img = tf.io.decode_jpeg(byte_img, channels=3)

        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, config.IMG_SIZE)
        # Scale image to be between 0 and 1
        img = img / 255.0

        return img
    except tf.errors.InvalidArgumentError as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

def create_dataset():
    anchor = tf.data.Dataset.list_files(config.ANC_PATH+'/*.jpg').take(config.NUM_IMAGES_TO_LOAD)
    positive = tf.data.Dataset.list_files(config.POS_PATH+'/*.jpg').take(config.NUM_IMAGES_TO_LOAD)
    negative = tf.data.Dataset.list_files(config.NEG_PATH+'/*.jpg').take(config.NUM_IMAGES_TO_LOAD)

    # Print the number of images in each category
    print(f"Number of anchor images: {tf.data.experimental.cardinality(anchor).numpy()}")
    print(f"Number of positive images: {tf.data.experimental.cardinality(positive).numpy()}")
    print(f"Number of negative images: {tf.data.experimental.cardinality(negative).numpy()}")

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=config.BUFFER_SIZE)

    train_data = data.take(round(len(data) * config.TRAIN_SPLIT))
    train_data = train_data.batch(config.BATCH_SIZE)
    train_data = train_data.prefetch(config.PREFETCH_BUFFER)

    test_data = data.skip(round(len(data) * config.TRAIN_SPLIT))
    test_data = test_data.take(round(len(data) * (1 - config.TRAIN_SPLIT)))
    test_data = test_data.batch(config.BATCH_SIZE)
    test_data = test_data.prefetch(config.PREFETCH_BUFFER)

    return train_data, test_data
# just for testing
def plot_images(ds):
    try:
        sample = next(iter(ds))
        if len(sample[0]) == 0:
            print("Dataset is empty")
            return

        n = min(len(sample[0]), 3)  # Display up to 3 images
        plt.figure(figsize=(5*n, 10))

        for i in range(n):
            plt.subplot(2, n, i+1)
            plt.imshow(sample[0][i])
            plt.title('Anchor')
            plt.axis('off')

            plt.subplot(2, n, n+i+1)
            plt.imshow(sample[1][i])
            plt.title('Positive' if sample[2][i] == 1 else 'Negative')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plot_images: {str(e)}")

if __name__ == "__main__":
    try:
        train_data, test_data = create_dataset()

        print("Number of train batches:", tf.data.experimental.cardinality(train_data).numpy())
        print("Number of test batches:", tf.data.experimental.cardinality(test_data).numpy())

        print("Visualizing training data:")
        plot_images(train_data)

        print("Visualizing test data:")
        plot_images(test_data)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
