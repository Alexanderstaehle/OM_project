import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt


def initialize_env(seed=42):
    """
    Sets environment variables and seeds to make model training deterministic
    """
    random.seed(seed)
    tf.random.set_seed(seed)


# Below code is is taken from https://medium.com/deep-learning-experiments/effect-of-batch-size-on-neural-net-training-c5ae8516e57
def load_dataset(dataset_name, shuffle_seed):
    """
    Loads the tensorflow datasets
    :param dataset_name: One of the following dataset names: https://www.tensorflow.org/datasets/catalog/overview
    :param shuffle_seed: Seed for shuffling
    :return: (raw_train, raw_validation), label_metadata
    """
    read_config = tfds.ReadConfig(shuffle_seed=shuffle_seed)
    (raw_test, raw_validation, raw_train), metadata = tfds.load(
        dataset_name,
        split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
        shuffle_files=True,
        with_info=True,
        as_supervised=True,
        read_config=read_config
    )
    # print('Training Data Summary')
    # summarize_dataset(raw_train)
    # print('\nValidation Data Summary')
    # summarize_dataset(raw_validation)
    return (raw_test, raw_validation, raw_train), metadata.features['label'].names


def summarize_dataset(tf_data):
    """
    Prints stats around no. of classes in the dataset
    :param tf_data: PrefetchDataset
    """
    label = np.array([l for _, l in tf_data])
    class_freq = np.array(np.unique(label, return_counts=True)).transpose()
    class_summary = {f[0]: (f[1], f[1] * 100 / len(label)) for f in class_freq}
    print('No. of examples: {count}'.format(count=len(label)))
    for k, v in class_summary.items():
        print('Class: {class_val} :::: Count: {count} :::: Percentage: {percent}'.format(
            class_val=k,
            count=v[0],
            percent=v[1]
        ))


def resize_image(image, label, img_size, normalize_pixel_values=True):
    """
    Resizes image
    :param image: Image
    :param label: Label
    :param img_size: Image size
    :param normalize_pixel_values: Whether to divide pixel values by 255
    :return:
    """
    image = tf.cast(image, tf.float32)
    if normalize_pixel_values:
        image = image / 255
    image = tf.image.resize(image, (img_size, img_size))
    return image, label


def show_image(image, label, label_names):
    """
    Shows image
    :param image: Image
    :param label: Label
    :param label_names: List containing label names
    :return:
    """
    plt.figure()
    plt.imshow(image)
    plt.title('Class: {class_value} :::: Class Name: {label_name}'.format(
        class_value=label,
        label_name=label_names[label] if len(label_names) > label else ''
    ))


def load_batched_and_resized_dataset(
        dataset_name,
        batch_size=32,
        img_size=128,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
        normalize_pixel_values=True
):
    """
    Resizes and normalizes images, caches them in memory, and divides them into batches
    :param dataset_name: One of the following dataset names: https://www.tensorflow.org/datasets/catalog/overview
    :param batch_size: Batch size
    :param img_size: Target image size, defaults to 128
    :param shuffle_buffer_size: Number of examples to load into buffer for shuffling, defaults to 1000
    :param shuffle_seed: Seed for shuffling, defaults to 0
    :param normalize_pixel_values: Whether to divide pixel values by 255.
    :return: train_batches, validation_batches
    """
    # Load dataset
    (raw_test, raw_validation, raw_train), label_names = load_dataset(dataset_name, shuffle_seed=shuffle_seed)

    # Resize images and normalize (divide by 255) if specified
    resize = lambda img, lbl: resize_image(img, lbl, img_size, normalize_pixel_values)
    train = raw_train.map(resize)
    validation = raw_validation.map(resize)
    test = raw_test.map(resize)

    # Cache data in memory
    train = train.cache()
    validation = validation.cache()
    test = test.cache()

    # Divide data into batches
    train_batches = train.shuffle(
        buffer_size=shuffle_buffer_size,
        seed=shuffle_seed,
        reshuffle_each_iteration=False,
    ).batch(batch_size)
    validation_batches = validation.batch(batch_size)
    test_batches = test.batch(batch_size)

    return train_batches, validation_batches, test_batches
