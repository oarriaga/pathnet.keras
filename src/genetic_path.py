import random
import numpy as np

class GeneticPath(object):
    def __init__(self, shape=(3,5), population_size=64,
                                max_num_active_paths=3):
        self.num_modules_per_layer, self.num_layers = shape
        self.max_num_active_paths = max_num_active_paths
        self.population_size = population_size
        self.population = self.generate_population()

    def generate_population(self):
        population = [self._generate_individual()
                for _ in range(self.population_size)]
        return population

    def _generate_individual(self):
        path = np.zeros(shape=(self.num_modules_per_layer, self.num_layers))
        return np.apply_along_axis(self._generate_valid_module,
                                            axis=0, arr=path)

    def _generate_valid_module(self, module):
        random_args = np.random.permutation(self.num_modules_per_layer)
        num_active_paths = np.random.randint(1, self.max_num_active_paths + 1)
        random_args = np.unravel_index(random_args[:num_active_paths],
                                        dims=self.num_modules_per_layer)
        module[random_args] = 1
        return module

    def sample_genotype_paths(self, num_genotypes=2):
        return random.sample(self.population, num_genotypes)

    def mutate_genotype_path(self, genotype_path):
        mutation_probabilities = np.random.random(size=(self.num_layers,
                                                self.num_active_paths))
        mutation_rate = 1.0 / (self.num_layers * self.num_active_paths)
        mutation_mask = mutation_probabilities < mutation_rate
        selected_paths = genotype_path[mutation_mask]
        selected_paths = selected_paths + np.random.randint(low=-2, high=2,
                                                size=selected_paths.shape)
        negative_mask = selected_paths < 0
        selected_paths[negative_mask] = (selected_paths[negative_mask] +
                                        self.num_modules_per_layer)
        positive_mask = selected_paths > self.num_modules_per_layer
        selected_paths[positive_mask] = (selected_paths[positive_mask] -
                                        (self.num_modules_per_layer - 1))
        genotype_path[mutation_mask] = selected_paths

    def overwrite(self, genotypes, fitnesses):
        win = genotypes[fitnesses.index(max(fitnesses))]
        lose = genotypes[fitnesses.index(min(fitnesses))]
        genotype = win.return_genotype()
        lose.overwrite(genotype)
        lose.mutate()

if __name__ == '__main__':
    genetic_paths = GeneticPath(shape=(5, 3))
    print(genetic_paths.population[0])
    path_1, path_2 = genetic_paths.sample_genotype_paths()
    print(path_1)
    print(path_2)

