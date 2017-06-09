import random
import numpy as np

class GeneticAgents(object):
    def __init__(self, shape=(3, 5), population_size=64,
                                max_num_active_paths=3):
        self.num_modules_per_layer, self.num_layers = shape
        self.max_num_active_paths = max_num_active_paths
        self.population_size = population_size
        self.population = self.generate_population()
        self.mutation_rate = 1.0 / (self.num_layers *
                                    self.max_num_active_paths)

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
        sampled_args = random.sample(list(range(self.population_size)),
                                                        num_genotypes)
        sampled_individuals = []
        for sampled_arg in sampled_args:
            sampled_individuals.append(self.population[sampled_arg])
        return sampled_individuals, sampled_args

    def mutate(self, genotype_path):
        mutation_probabilities = np.random.random(size=(
                self.num_modules_per_layer, self.num_layers))
        inside_path_mask = genotype_path == 1
        mutation_mask = self.mutation_rate > mutation_probabilities
        mutation_mask = np.logical_and(inside_path_mask, mutation_mask)
        genotype_path[mutation_mask] = 0
        y_mutated_args, x_mutated_args = np.where(mutation_mask)
        y_mutation = np.random.randint(-2 , 2, size=y_mutated_args.shape)
        y_mutated_args = y_mutated_args + y_mutation
        max_module_arg = self.num_modules_per_layer - 1
        y_mutated_args = np.clip(y_mutated_args, 0, max_module_arg)
        mutated_args = (y_mutated_args, x_mutated_args)
        genotype_path[mutated_args] = 1
        return genotype_path

    def overwrite(self, genotype_args, fitness_values):
        genotype_arg_1, genotype_arg_2 = genotype_args
        fitness_value_1, fitness_value_2 = fitness_values
        if fitness_value_1 >= fitness_value_2:
            genotype_path_1 = self.population[genotype_arg_1].copy()
            self.population[genotype_arg_2] = self.mutate(genotype_path_1)
            return genotype_path_1
        else:
            genotype_path_2 = self.population[genotype_arg_2].copy()
            self.population[genotype_arg_1] = self.mutate(genotype_path_2)
            return genotype_path_2

if __name__ == '__main__':
    genetic_agents = GeneticAgents(shape=(5, 3))
    paths, path_args = genetic_agents.sample_genotype_paths()
    path_1, path_2 = paths
    loss = (100, 1)
    print(loss)
    genetic_agents.overwrite(path_args, loss)
    print('overwritten_path_1 (winner): \n',
            genetic_agents.population[path_args[0]])
    print('overwritten_path_2 (loser): \n',
            genetic_agents.population[path_args[1]])
    print(genetic_agents.population[path_args[0]] == genetic_agents.population[path_args[1]])
