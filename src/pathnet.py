from keras.layers import Dense
from keras.layers import Input
from layers import ReduceSum
from keras.models import Model
import numpy as np

class PathNet(object):
    def __init__(self, shape=(3,5), population_size=64,
                input_size=10, output_size=20, num_neurons_per_module=20):
        self.num_modules_per_layer, self.num_layers = shape
        self.population_size = population_size
        self.input_layer = Input(shape=(input_size,))
        self.output_size = output_size
        self.output_layer = Dense(self.output_size)
        self.num_neurons_per_module = num_neurons_per_module
        self.pathnet = self._instantiate_pathnet()

    def _instantiate_pathnet(self):
        modules = []
        for module_arg in range(self.num_modules_per_layer):
            layers = []
            for layer_arg in range(self.num_layers):
                name = 'module_' + str(module_arg) + str(layer_arg)
                layers.append(Dense(self.num_neurons_per_module, name=name))
            modules.append(layers)
        return np.asarray(modules)

    def compile_path(self, individual):
        reduced_sum_modules = []
        for layer_arg in range(self.num_layers):
            layer_paths = []
            for module_arg in range(self.num_modules_per_layer):
                if individual[module_arg, layer_arg] == 1:
                    if layer_arg == 0:
                        layer_paths.append(self.pathnet[module_arg, layer_arg](
                                                            self.input_layer))
                    else:
                        layer_paths.append(self.pathnet[module_arg, layer_arg](
                                        reduced_sum_modules[layer_arg - 1]))
            reduced_sum_modules.append(ReduceSum(axis=-1)(layer_paths))

        predictions = self.output_layer(reduced_sum_modules[-1])
        model = Model(inputs=self.input_layer, outputs=predictions)
        return model

if __name__ == '__main__':
    from keras.utils import plot_model
    from genetic_agents import GeneticAgents



    num_layers = 5
    num_modules_per_layer = 10
    pathnet = PathNet(shape=(num_modules_per_layer, num_layers))
    individual = np.ones(shape=(num_modules_per_layer, num_layers))
    full_model = pathnet.compile_path(individual)
    plot_model(full_model, to_file='../images/full_pathnet.png')
    random_individual = np.random.randint(0, 2,
                        size=(num_modules_per_layer, num_layers))
    path_model = pathnet.compile_path(random_individual)
    plot_model(path_model, to_file='../images/random_pathnet.png')

    genetic_agents = GeneticAgents(shape=(num_modules_per_layer, num_layers))
    paths, path_args = genetic_agents.sample_genotype_paths()
    path_1, path_2 = paths
    individual_path = pathnet.compile_path(path_1)
    plot_model(individual_path, to_file='../images/agent_path.png')

