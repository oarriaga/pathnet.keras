import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

def normalize_image(image_array):
    return image_array / 255.

def display_image(image_array, image_class, cmap=None):
    image_array = np.squeeze(image_array)
    image_array = image_array.astype('uint8')
    plt.imshow(image_array, cmap)
    image_class = str(image_class)
    plt.title('ground truth class: ' + image_class)
    plt.show()

def split_data(train_data, validation_split=.2):
    num_train = int(round((1 - validation_split) * len(train_data[0])))
    train_images, train_classes = train_data
    train_images = train_images[:num_train]
    train_classes = train_classes[:num_train]
    validation_images = train_data[num_train:]
    validation_classes = train_data[num_train:]
    train_data = (train_images, train_classes)
    validation_data = (validation_images, validation_classes)
    return train_data, validation_data

def save_layer_weights(model, save_path):
    for layer in model.layers:
        file_path = save_path + layer.name + '.p'
        pickle.dump(layer.get_weights(), open(file_path, 'wb'))

def load_layer_weights(model, save_path):
    file_paths = glob.glob(save_path + '*.p')
    weight_filenames = [os.path.basename(path)[:-2] for path in file_paths]
    for layer in model.layers:
        if layer.name in weight_filenames:
            weights_path = save_path + layer.name + '.p'
            weights = pickle.load(open(weights_path, 'rb'))
            layer.set_weights(weights)
