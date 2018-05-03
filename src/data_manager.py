import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataManager(object):

    def __init__(self, filepath, split_length=10, num_classes=None):

        self.filepath = filepath
        self.split_length = split_length
        self.num_classes = num_classes

    def _preprocess_data(self, data):
        return preprocessing.scale(data, axis=0)

    def _infer_num_classes(self, data):
        num_classes = data.iloc[:, -1].unique().shape[0]
        return num_classes

    def _load_data(self, filepath):
        data = pd.read_csv(self.filepath, delimiter=',', header=None)
        if self.num_classes is None:
            self.num_classes = self._infer_num_classes(data)
        data = data.iloc[:, :9].as_matrix()
        return data

    def _split_time_series(self, time_series, split_length=300):
        """ Transform data into time series
        Arguments:
            data: numpy.array, array of shape (n_samples_X, n_features)
            series_length: int, length of a single series.
        Returns:
            numpy.array (n_samples/series_length, series_length, n_features)
        """
        series_length, num_channels = time_series.shape
        remainder = series_length % split_length
        if remainder != 0:
            time_series = time_series[:-remainder]
        num_splits = int(series_length/split_length)
        return time_series.reshape(num_splits, split_length, num_channels)

    def _get_class_time_series(self, class_arg, data):
        series_length = data.shape[0]
        start_arg = int((series_length / self.num_classes) * class_arg)
        end_arg = int((series_length / self.num_classes) * (class_arg + 1))
        class_time_series = data[start_arg:end_arg]
        return class_time_series

    def _make_one_hot_features(self, class_arg, class_features):
        num_samples = class_features.shape[0]
        class_targets = np.zeros(shape=(num_samples, self.num_classes))
        class_targets[:, class_arg] = 1
        return class_targets

    def _change_data_for_regression(self, features):
        class_features = features[:, :, :6]
        class_targets = features[:, :, 6:]
        return class_features, class_targets

    def _randomize_data(self, class_features, class_targets, seed):
        np.random.seed(seed)
        num_samples = class_features.shape[0]
        random_sample_args = np.arange(num_samples)
        np.random.shuffle(random_sample_args)
        class_features = class_features[random_sample_args]
        class_targets = class_targets[random_sample_args]
        return class_features, class_targets

    def load_data(self, train_split=.6, val_split=.2, regression=False,
                  randomize=False, classes=False, flatten=False, seed=7):

        data = self._load_data(self.filepath)
        data = self._preprocess_data(data)

        train_features, val_features, test_features = [], [], []
        train_targets, val_targets, test_targets = [], [], []
        for class_arg in range(self.num_classes):
            class_time_series = self._get_class_time_series(class_arg, data)
            class_features = self._split_time_series(
                    class_time_series, self.split_length)

            if regression:
                class_data = self._change_data_for_regression(class_features)
                class_features, class_targets = class_data

            else:
                class_targets = self._make_one_hot_features(
                        class_arg, class_features)

            if randomize:
                class_features, class_targets = self._randomize_data(
                        class_features, class_targets, seed)

            train_split_arg = int(len(class_targets) * train_split)
            train_features.append(class_features[:train_split_arg])
            train_targets.append(class_targets[:train_split_arg])

            val_test_features = class_features[train_split_arg:]
            val_test_targets = class_targets[train_split_arg:]

            relative_val_split = val_split / (1 - train_split)
            relative_val_split_arg = (int(len(val_test_targets) *
                                      relative_val_split))

            val_features.append(val_test_features[:relative_val_split_arg])
            val_targets.append(val_test_targets[:relative_val_split_arg])

            test_features.append(val_test_features[relative_val_split_arg:])
            test_targets.append(val_test_targets[relative_val_split_arg:])

        if classes:
            train_features = np.concatenate(train_features, 0)
            train_targets = np.concatenate(train_targets, 0)

            val_features = np.concatenate(val_features, 0)
            val_targets = np.concatenate(val_targets, 0)

            test_features = np.concatenate(test_features, 0)
            test_targets = np.concatenate(test_targets, 0)

        train_data = (train_features, train_targets)
        val_data = (val_features, val_targets)
        test_data = (test_features, test_targets)

        return train_data, val_data, test_data


if __name__ == "__main__":
    filepath = '../datasets/3_classes_small_new.csv'
    data_manager = DataManager(filepath, split_length=20)
    train, val, test = data_manager.load_data(.6, .2, True, False, False)
    print(train[0].shape, train[1].shape)
