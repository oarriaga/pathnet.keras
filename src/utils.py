import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

def normalize_images(image_array):
    return image_array.astype('float32') / 255.

def _add_salt_and_pepper(image_array, probability=.5):
    image_array = np.squeeze(image_array)
    uniform_values = np.random.rand(*image_array.shape)
    spiced_image = image_array.copy()
    spiced_image = spiced_image.astype('float32')
    peppery_mask = uniform_values < probability/2.
    spiced_image[peppery_mask] = np.min(image_array)
    salty_mask = uniform_values > (1 - probability/2.)
    spiced_image[salty_mask] = np.max(image_array)
    return spiced_image

def spice_up_images(images):
    num_samples = len(images)
    for sample_arg in range(num_samples):
        images[sample_arg] = _add_salt_and_pepper(images[sample_arg])
    return images

def split_data(train_data, validation_split=.2):
    num_train = int(round((1 - validation_split) * len(train_data[0])))
    train_images, train_classes = train_data
    validation_images = train_images[num_train:]
    validation_classes = train_classes[num_train:]
    train_images = train_images[:num_train]
    train_classes = train_classes[:num_train]
    train_data = (train_images, train_classes)
    validation_data = (validation_images, validation_classes)
    return train_data, validation_data

def display_image(image_array, image_class, cmap=None):
    image_array = np.squeeze(image_array)
    image_array = image_array.astype('uint8')
    plt.imshow(image_array, cmap)
    image_class = str(image_class)
    plt.title('ground truth class: ' + image_class)
    plt.show()

def from_path_to_names(paths):
    if (len(paths) is 0) or (paths is None):
        return []
    frozen_path_names = []
    for path in paths:
        path_coordinates = list(zip(*[coordinates.tolist()
                        for coordinates in np.where(path)]))
        names = ['module_' + str(args[0]) + str(args[1])
                        for args in path_coordinates]
        frozen_path_names + frozen_path_names + names
    return list(set(frozen_path_names))

def save_layer_weights(model, save_path, frozen_paths=[]):
    frozen_path_names = from_path_to_names(frozen_paths)
    for layer in model.layers:
        layer_name = layer.name
        if ('reduce' in layer_name) or (layer_name in frozen_path_names):
            continue
        file_path = save_path + layer.name + '.p'
        pickle.dump(layer.get_weights(), open(file_path, 'wb'))

def load_layer_weights(model, save_path):
    file_paths = glob.glob(save_path + '*.p')
    if file_paths is None:
        print('file_paths not found, continuing without loading weights...')
        return
    weight_filenames = [os.path.basename(path)[:-2] for path in file_paths]
    for layer in model.layers:
        if layer.name in weight_filenames:
            weights_path = save_path + layer.name + '.p'
            weights = pickle.load(open(weights_path, 'rb'))
            layer.set_weights(weights)

def reset_weights(frozen_paths, save_path):
    frozen_path_names = from_path_to_names(frozen_paths)
    file_paths = glob.glob(save_path + '*.p')
    weight_filenames = [os.path.basename(path)[:-2] for path in file_paths]
    for weight_filename in weight_filenames:
        if weight_filename in frozen_path_names:
            continue
        else:
            os.remove(save_path + weight_filename + '.p')

def to_categorical(data):
    arg_classes = np.unique(data).tolist()
    arg_classes.sort()
    num_classes = len(arg_classes)
    label_to_arg = dict(zip(arg_classes, list(range(num_classes))))
    num_samples = len(data)
    categorical_data = np.zeros(shape=(num_samples, num_classes))
    for sample_arg in range(num_samples):
        label = data[sample_arg]
        data_arg = label_to_arg[label]
        categorical_data[sample_arg, data_arg] = 1
    return categorical_data

def flatten(images):
    num_samples = len(images)
    return images.reshape(num_samples, -1)

def shuffle(input_data, output_classes):
    num_samples = len(input_data)
    random_args = np.random.permutation(np.arange(num_samples))
    input_data = input_data[random_args]
    output_classes = output_classes[random_args]
    return input_data, output_classes

if __name__ == "__main__":
    class_data = np.array([7, 7, 6, 6 ,6 ,5 ,6 ,6])
    print(class_data)
    categorical_data = to_categorical(class_data)
    print(categorical_data)
