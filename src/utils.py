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





