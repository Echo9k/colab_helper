from os import path
from pathlib import Path
import tensorflow as tf
from numpy import array

from .GetData import GetData


def get_label(file_path, class_names):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img, target_size):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [target_size[0], target_size[1]])


class Preproccessing(GetData):
    def __init__(self, **kwargs):
        super().__init__()
        self.image_count = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def process_path(self, file_path):
        label = get_label(file_path, self.class_names)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img, self.target_size)
        return img, label

    @staticmethod
    def configure_for_performance(dataset, batch_size):
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def dataset_preprocess(self, img_format='jpg', validation_split: int = 0.3):
        data_dir = Path(self.img_directory)

        # Count files
        self.image_count = len(list(data_dir.glob(f'*/*.{img_format}')))
        print(f"Total training images: {self.image_count}")

        # List datasets
        list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))
        list_ds = list_ds.shuffle(self.image_count, reshuffle_each_iteration=False)

        # Get class names in directory
        class_names = array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
        print(class_names)

        # Split into train and valid
        val_size = int(self.image_count * validation_split)
        train_ds = list_ds.skip(val_size)

        print(f"Train samples: {tf.data.experimental.cardinality(train_ds).numpy()}")

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        return train_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
