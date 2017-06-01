import random
import numpy as np

class GeneticPath(object):
    def __init__(self, shape=(3, 5), population_size=64,
                                max_num_active_paths=3):
        self.num_modules_per_layer, self.num_layers = shape
        self.max_num_active_paths = max_num_active_paths
        self.population_size = population_size
        self.population = self.generate_population()
        self.mutation_rate = 1.0 / (self.num_layers * self.max_num_active_paths)

    def generate_population(self):
        population = [self._generate_genotype_path()
                for _ in range(self.population_size)]
        return population

    def _generate_genotype_path(self):
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

    def mutate(self, genotype_path):
        mutation_probabilities = np.random.random(size=(
                self.num_modules_per_layer, self.num_layers))
        inside_path_mask = genotype_path == 1
        mutation_mask = self.mutation_rate > mutation_probabilities
        # mutation is only applied to the preexisting paths.
        mutation_mask = np.logical_and(inside_path_mask, mutation_mask)
        # restart the to-be mutated genes.
        genotype_path[mutation_mask] = 0
        x_mutated_args, y_mutated_args = np.where(mutation_mask)
        x_mutation = np.random.randint(-2 , 2, size=x_mutated_args.shape)
        y_mutation = np.random.randint(-2 , 2, size=y_mutated_args.shape)
        x_mutated_args = x_mutated_args + x_mutation
        y_mutated_args = y_mutated_args + y_mutation
        max_module_arg = self.num_modules_per_layer - 1
        x_mutated_args = np.clip(x_mutated_args, 0, max_module_arg)
        y_mutated_args = np.clip(y_mutated_args, 0, max_module_arg)
        mutated_args = (x_mutated_args, y_mutated_args)
        genotype_path[mutated_args] = 1
        return genotype_path

    def mutate_genotype_path(self, genotype_path):
        mutation_probabilities = np.random.random(size=(self.num_layers,
                                                self.num_active_paths))
        mutation_mask = self.mutation_rate > mutation_probabilities
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
    path_1, path_2 = genetic_paths.sample_genotype_paths()
    print('path_1 \n', path_1)
    #print('path_2 \n', path_2)
    mutated_path_1 = genetic_paths.mutate(path_1)
    print('mutated_path_1 \n', mutated_path_1)


