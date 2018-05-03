import numpy as np


class GeneticAgents(object):
    def __init__(self, shape=(3, 5), population_size=64,
                 max_num_active_paths=3, seed=None):

        np.random.seed(seed)
        self.num_modules, self.num_layers = shape
        self.max_num_active_paths = max_num_active_paths
        self.population_size = population_size
        self.mutation_rate = 1. / (self.num_layers * self.max_num_active_paths)
        self.population = self.generate_population()

    def generate_population(self):
        return [self._generate_path() for _ in range(self.population_size)]

    def _generate_path(self):
        path = np.zeros(shape=(self.num_modules, self.num_layers))
        return np.apply_along_axis(self._generate_module, 0, path)

    def _generate_module(self, module):
        num_active_paths = np.random.randint(1, self.max_num_active_paths + 1)
        random_args = np.random.choice(self.num_modules, num_active_paths)
        module[random_args] = 1
        return module

    def sample_genotype_paths(self, num_genotypes=2):
        sampled_args = np.random.choice(self.population_size, num_genotypes)
        sampled_individuals = [self.population[arg] for arg in sampled_args]
        return sampled_individuals, sampled_args

    def mutate(self, path, min_value=-2, max_value=2):
        path_mask = path == 1
        shape = (self.num_modules, self.num_layers)
        mutation_mask = self.mutation_rate > np.random.random(shape)
        mutation_mask = np.logical_and(path_mask, mutation_mask)
        path[mutation_mask] = 0
        mutated_module_args, mutated_layer_args = np.where(mutation_mask)
        num_mutations = np.sum(mutation_mask)
        mutations = np.random.randint(min_value, max_value, num_mutations)
        mutated_module_args = mutated_module_args + mutations
        max_module_arg = self.num_modules - 1
        mutated_module_args = np.clip(mutated_module_args, 0, max_module_arg)
        path[(mutated_module_args, mutated_layer_args)] = 1
        return path

    def overwrite(self, path_args, fitness_values):
        path_arg_1, path_arg_2 = path_args
        fitness_value_1, fitness_value_2 = fitness_values
        if fitness_value_1 >= fitness_value_2:
            path_1 = self.population[path_arg_1].copy()
            self.population[path_arg_2] = self.mutate(path_1)
            return path_1
        else:
            path_2 = self.population[path_arg_2].copy()
            self.population[path_arg_1] = self.mutate(path_2)
            return path_2


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
    print(genetic_agents.population[path_args[0]] ==
          genetic_agents.population[path_args[1]])
