from keras.layers import Dense
from keras.layers import Input
from layers import ReduceSum
from keras.models import Model
import numpy as np

class PathNet(object):
    def __init__(self, num_layers=3, num_modules_per_layer=10,
                    population_size=64, num_active_paths=3):
        self.num_layers = num_layers
        self.num_modules_per_layer = num_modules_per_layer
        self.population_size = population_size
        self.num_active_paths = num_active_paths
        self.output_size = (10)
        self.input_layer = Input(shape=(20,))
        self.output_layer = Dense(self.output_size)
        self.num_neurons_per_module = 20
        #self.module_paths = self._calculate_all_paths()
        self.pathnet = self._instantiate_pathnet()

    def _instantiate_pathnet(self):
        layers = []
        for layer_arg in range(self.num_layers):
            modules = []
            for module_arg in range(self.num_modules_per_layer):
                modules.append(Dense(self.num_neurons_per_module))
            layers.append(modules)
        return np.asarray(layers)

    def compile_path(self, individual):
        """ pathnet should be a numpy array of size
        num_layers x num_modules_per_layer
        """
        #path_mask = individual == True
        #selected_path = pathnet[path_mask]
        num_layers, num_modules_per_layer = individual.shape
        reduced_sum_modules = []
        for layer_arg in range(num_layers):
            layer_paths = []
            for module_arg in range(num_modules_per_layer):
                if individual[layer_arg, module_arg]:
                    if layer_arg == 0:
                        layer_paths.append(self.pathnet[layer_arg, module_arg](
                                                            self.input_layer))
                    else:
                        layer_paths.append(self.pathnet[layer_arg, module_arg](
                                        reduced_sum_modules[layer_arg - 1]))
            reduced_sum_modules.append(ReduceSum(axis=-1)(layer_paths))

        predictions = self.output_layer(reduced_sum_modules[-1])
        model = Model(inputs=self.input_layer, outputs=predictions)
        return model

if __name__ == '__main__':
    from keras.utils import plot_model

    num_layers = 5
    num_modules_per_layer = 10
    pathnet = PathNet(num_layers, num_modules_per_layer)
    individual = np.ones(shape=(num_layers, num_modules_per_layer))
    model = pathnet.compile_path(individual)
    plot_model(model, to_file='../images/random_pathnet.png')



