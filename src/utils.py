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





