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
    def load(self, dataset_name):
        self.dataset_downloader = DataDownloader(self.dataset_path_prefix,
                                                    dataset_name)
        if dataset_name == 'mnist':
            train_data, test_data = mnist.load_data()
            return (train_data, test_data)
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

if __name__ == '__main__':
    data_manager = DataManager()
    print('Available datasets: ', data_manager.available_datasets)

    train_data, test_data = data_manager.load('mnist')
    train_images, train_classes = train_data
    display_image(train_images[0], train_classes[0], cmap='gray')

    train_data, test_data = data_manager.load('svhn')
    train_images, train_classes = train_data
    display_image(train_images[0], train_classes[0], cmap='gray')

