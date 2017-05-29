import random
import numpy as np

class GeneticPath(object):
    def __init__(self, num_layers, num_modules_per_layer,
                population_size, num_active_paths=3):
        self.num_layers = num_layers
        self.num_modules_per_layer = num_modules_per_layer
        self.num_active_paths = num_active_paths
        self.population_size = population_size
        self.genotype_paths = []
        for population_arg in range(population_size):
            genotype_path = np.random.randint(low=0,
                    high=self.num_modules_per_layer,
                    size=(self.num_layers, self.population_size))
            self.genotype_paths.append(genotype_path)

    def sample_genotype_paths(self, num_genotypes=2):
        return random.sample(self.genotype_paths, num_genotypes)

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
    genetic_paths = GeneticPath(5, 3, 3, 10)
    print(genetic_paths.population)

