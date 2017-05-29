import itertools

from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
import keras.backend as K

class PathNet(object):
    def __init__(self, num_layers=3, num_modules_per_layer=10,
                    population_size=64, num_active_paths=3):
        self.num_layers = num_layers
        self.num_modules_per_layer = num_modules_per_layer
        self.population_size = population_size
        self.num_active_paths = num_active_paths
        self.input_size = (48, 48)
        self.input_layer = Input(shape=(784,))
        self.output_layer = Dense(self.output_size)
        self.num_neurons_per_module = 20
        self.module_paths = self._calculate_all_paths()
        self.layers = self._instantiate_pathnet()

    def _reduce_sum(tensors):
        tensor_sum = K.zeros_like(tensors[0])
        for tensor in tensors:
            tensor_sum = tensor_sum + tensor
        return

    def _instantiate_pathnet(self):
        layers = []
        for layer_arg in range(self.num_layers):
            modules = []
            for module_arg in range(self.num_modules_per_layer):
                modules.append(Dense(self.num_neurons_per_module))
            layers.append(modules)
        return layers

    def compile_path(self, individual):
        """ pathnet should be a numpy array of size
        num_layers x num_modules_per_layer
        """
        num_layers, num_modules_per_layer = individual.shape
        for layer_arg in range(num_layers):
            for module_arg in range(num_modules_per_layer):
                if individual[layer_arg, module_arg]:
                    if layer_arg == 0:
                        path = self.layers[layer_arg, module_arg](
                                                        self.input_layer)
                    else:
                        path = self.layers[layer_arg, module_arg](
                                        self.sum_modules[layer_arg])


        path = self.output_layer(path)
        model = Model(inputs=self.input_layer, output=path)
        return model

if __name__ == '__main__':
    pathnet = PathNet()

