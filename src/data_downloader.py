import os
from urllib.request import urlopen

class DataDownloader(object):
    def __init__(self, datasets_path, dataset_name):
        self.datasets_path = datasets_path

    def get(self, dataset_name):
        if dataset_name == 'svhn':
            train_data_url = ('http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
            test_data_url = ('http://ufldl.stanford.edu/housenumbers/test_32x32.mat')
            urls = [train_data_url, test_data_url]
            for url in urls:
                self._download(url, dataset_name)

    def _download(self, url, dataset_name):
        url_file = urlopen(url)
        dataset_directory = self.datasets_path + dataset_name + '/'
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)
        data_type_directory = os.path.basename(url)
        filename = os.path.join(dataset_directory, data_type_directory)
        if not os.path.exists(filename):
            print("Downloading data from {}".format(url))
            with open(filename, 'wb') as local_file:
                local_file.write(url_file.read())
        else:
            print('Found dataset type %s' % data_type_directory)
