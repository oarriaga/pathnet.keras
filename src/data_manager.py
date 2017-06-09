import os

import scipy.io
import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar10

from data_downloader import DataDownloader
from utils import display_image

class DataManager(object):
    """Class for loading the different datasets used in pathnets"""
    def __init__(self, dataset_path_prefix='../datasets/'):
        if not os.path.isdir(dataset_path_prefix):
            os.makedirs(dataset_path_prefix)
        self.available_datasets = ('mnist', 'cifar10', 'svhn')
        self.dataset_path_prefix = dataset_path_prefix
        self.dataset_downloader = None

    def load(self, dataset_name, class_args=None):
        self.dataset_downloader = DataDownloader(self.dataset_path_prefix,
                                                            dataset_name)
        if dataset_name == 'mnist':
            train_data, test_data = mnist.load_data()
            if class_args is not None:
                train_data, test_data = self._filter_classes((train_data,
                                                 test_data), class_args)
                train_data, test_data = (train_data, test_data)
            return self._shuffle((train_data, test_data))
        elif dataset_name == 'cifar10':
            train_data, test_data = cifar10.load_data()

        elif dataset_name == 'svhn':
            if dataset_name not in os.listdir(self.dataset_path_prefix):
                self.dataset_downloader.get(dataset_name)
            else:
                print('Found dataset %s directory skipping download'
                                                    % dataset_name)
            dataset_path = self.dataset_path_prefix + dataset_name + '/'
            file_names = os.listdir(dataset_path)
            for file_name in file_names:
                if 'train' in os.path.basename(file_name):
                    data = scipy.io.loadmat(dataset_path + file_name)
                    input_train_data = np.rollaxis(data['X'], -1, 0)
                    output_train_data = np.squeeze(data['y'])
                    train_data = (input_train_data, output_train_data)
                elif 'test' in os.path.basename(file_name):
                    data = scipy.io.loadmat(dataset_path + file_name)
                    input_test_data = np.rollaxis(data['X'], -1, 0)
                    output_test_data = np.squeeze(data['y'])
                    test_data = (input_test_data, output_test_data)
            return (train_data, test_data)
        elif dataset_name not in self.available_datasets :
            raise Exception('Dataset name "%s" not valid' % dataset_name)

    def _filter_classes(self, data, class_args):
        train_data, test_data = data
        train_images, train_classes = train_data
        test_images, test_classes = test_data
        filtered_test_images = []
        filtered_train_images = []
        filtered_test_classes = []
        filtered_train_classes = []
        for class_arg in class_args:
            test_classes_mask = test_classes == class_arg
            train_classes_mask = train_classes == class_arg
            selected_test_images = test_images[test_classes_mask]
            selected_train_images = train_images[train_classes_mask]
            selected_test_classes = test_classes[test_classes_mask]
            selected_train_classes = train_classes[train_classes_mask]
            filtered_test_images.append(selected_test_images)
            filtered_train_images.append(selected_train_images)
            filtered_test_classes.append(selected_test_classes)
            filtered_train_classes.append(selected_train_classes)
        test_images = np.concatenate(filtered_test_images, axis=0)
        train_images = np.concatenate(filtered_train_images, axis=0)
        test_classes = np.concatenate(filtered_test_classes, axis=0)
        train_classes = np.concatenate(filtered_train_classes, axis=0)
        test_data = (test_images, test_classes)
        train_data = (train_images, train_classes)
        return (train_data, test_data)

    def _shuffle(self, data):
        train_data, test_data = data
        test_images, test_classes = test_data
        train_images, train_classes = train_data
        random_test_args = np.random.permutation(len(test_images))
        random_train_args = np.random.permutation(len(train_images))
        test_images = test_images[random_test_args]
        test_classes = test_classes[random_test_args]
        train_images = train_images[random_train_args]
        train_classes = train_classes[random_train_args]
        test_data = (test_images, test_classes)
        train_data = (train_images, train_classes)
        return (train_data, test_data)

if __name__ == '__main__':
    from utils import shuffle
    from utils import add_salt_and_pepper

    data_manager = DataManager()
    print('Available datasets: ', data_manager.available_datasets)

    train_data, test_data = data_manager.load('mnist', class_args=[0, 2])
    train_images, train_classes = train_data
    train_images, train_classes = shuffle(train_images, train_classes)
    for image_arg in range(10):
        image_array = add_salt_and_pepper(train_images[image_arg])
        display_image(image_array, train_classes[image_arg], cmap='gray')

    #train_data, test_data = data_manager.load('svhn')
    #train_images, train_classes = train_data
    #display_image(train_images[0], train_classes[0], cmap='gray')

