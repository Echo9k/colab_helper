from os import mkdir, walk, path
from typing import Tuple, Optional, Generator, Dict, List
import tensorflow as tf
from .ColabHelper import ColabHelper


def _list_folders(img_directory=None) -> List:
    """Get's all the folders on the img_directory.

    :param img_directory: The folder names inside a img_directory
    :return: List of folders in the given directory
    """
    try:
        _, directories, _ = next(walk(img_directory))
        return ([], directories)[len(directories) > 0]  # return directory names
    except StopIteration:
        'The attribute img_directory should be relative./"folder" or complete /.../"folder"'
    except NameError:
        "The specified directory doesn't exist yet"


def _prevent_duplicates(dir_url, class_names, img_directory) -> Dict:
    """
    From the dir{folder, url} grabs the requested folders and filters out the existent directories.
    :param dir_url:
    :param class_names:
    :param img_directory:
    :return: dir{folder: URL}
    """
    classes_dir = set(dir_url)
    list_labels = _list_folders(img_directory)
    classes_in_directory = (set(), set(list_labels))[len(list_labels) > 0]

    if class_names is not None:
        requested_classes = set(class_names)
        to_download = classes_dir.intersection(requested_classes) - classes_in_directory  # Requested but not in dir
    else:
        to_download = classes_dir - classes_in_directory  # Requested but not in dir

    if len(classes_in_directory) > 0:
        print(f"Folders found {classes_in_directory}")

    print(f'Downloading: {to_download}.')
    return {i: dir_url.get(i) for i in to_download}


def _img_dir(img_directory):
    if not path.exists(img_directory):
        mkdir(img_directory)
    return img_directory


def img_generator(validation_split=0.3,
                  samplewise_center=True,
                  samplewise_std_normalization=True, *,
                  featurewise_center=False,
                  featurewise_std_normalization=False,
                  zca_whitening=False,
                  fill_mode='reflect',
                  horizontal_flip=True,
                  vertical_flip=True,
                  additional_preprocessing=None):
    img_gen_params = {'featurewise_center': featurewise_center,
                      'samplewise_center': samplewise_center,
                      'featurewise_std_normalization': featurewise_std_normalization,
                      'samplewise_std_normalization': samplewise_std_normalization,
                      'zca_whitening': zca_whitening,
                      'fill_mode': fill_mode,
                      'horizontal_flip': horizontal_flip,
                      'vertical_flip': vertical_flip,
                      'validation_split': validation_split,
                      'additional_preprocessing': additional_preprocessing
                      }
    return tf.keras.preprocessing.image.ImageDataGenerator(**img_gen_params)


class GetData(ColabHelper):
    def __init__(self, dir_url: Dict[str, str] = None, class_names: [List] = None,
                 img_directory: str = './image_files'):
        super().__init__(dir_url, class_names, img_directory)
        self.dir_url = dir_url
        self.img_directory = _img_dir(img_directory)
        self.class_names = (class_names, _list_folders(img_directory))[class_names is None]
        self.batch_size = None

    def download_unzip(self, archive_format='auto', extract=True, as_generator=False) -> None or Generator:
        """
        Downloads data from the dir_url of the form {category, url}.
        Stores each folder under img_directory/category/

        PARAMETERS:
        :param extract:
        :param archive_format:Archive format to try for extracting the file. Options are 'auto', 'tar', 'zip', and None.
                        'tar' includes tar, tar.gz, and tar.bz files.
                        The default 'auto' corresponds to ['tar', 'zip'].
                        None or an empty list will return no matches found.
        :param as_generator: [default=False] Changes the behavior of the downloader.
         If set to false it will automatically download all the folders from  "self.dir_url"
         and store them in folders with the keys of that directory.

        :return: None, or a generator if asGenerator is set to True.
        """
        to_download = _prevent_duplicates(self.dir_url, self.class_names, self.img_directory)
        try:
            if not as_generator:
                for key, url in to_download.items():
                    tf.keras.utils.get_file(key, url,
                                            extract=extract,
                                            archive_format=archive_format,
                                            cache_subdir=self.img_directory + key)
            else:
                return (tf.keras.utils.get_file(key, url) for key, url in self.dir_url.items() if key in to_download)
        except TypeError:
            print(f"All the requested folders already exist on '{self.img_directory}'")

    def img_batch(self, batch_size: Optional[int] = 32,
                  target_size: Optional[Tuple] = (256, 256),
                  class_names: Optional[List] = None,
                  subset: Optional[str] = 'training', *,
                  class_mode: Optional[str] = 'categorical',
                  image_data_generator='default') -> tf.keras.preprocessing.image.DirectoryIterator:

        """ generates a batch of preprocessed data using the image_data_generator
        :param batch_size: size of the batches of data (default: 32)
        :param target_size: size of the output images.
        :param class_names:
        :param subset: Options "training", "validation".
        :param class_mode: Type of classification for this flow of data
<           - binary:if there are only two classes
            - categorical: categorical targets,
            - sparse: integer targets,
            - input: targets are images identical to input images (mainly used to work with autoencoders),
            - None: no targets get yielded (only input images are yielded).
        :param image_data_generator: preconfigured tf.keras.preprocessing.image.ImageDataGenerator

        :return DirectoryIterator"""

        image_data_generator = img_generator() if (image_data_generator == 'default') else image_data_generator
        assert isinstance(image_data_generator, tf.keras.preprocessing.image.ImageDataGenerator)

        img_dir_params = dict(directory=self.img_directory, image_data_generator=image_data_generator,
                              target_size=target_size, color_mode='rgb', classes=class_names,
                              class_mode=class_mode, batch_size=batch_size, shuffle=True)
        print(subset + ':')
        return tf.keras.preprocessing.image.DirectoryIterator(**img_dir_params, subset=subset)
