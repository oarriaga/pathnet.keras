from data_manager import DataManager
from genetic_agents import GeneticAgents
from pathnet import PathNet
from train_manager import TaskManager

filepath = '../datasets/3_classes_small_new.csv'
num_layers = 3
num_modules = 10
shape = (num_modules, num_layers)
population_size = 32
max_num_active_paths = 3
num_module_neurons = 10
split_length = 20
input_size = split_length * 6
output_size = split_length * 3
num_tasks = 3
max_num_genetic_epochs = 500
num_path_epochs = 100
num_samples_per_epoch = 16*50

data_manager = DataManager(filepath, split_length=20)
data_splits = data_manager.load_data(
        .6, .2, regression=True, randomize=False, classes=False)

genetic_agents = GeneticAgents(shape, population_size, max_num_active_paths)

pathnet = PathNet(shape, population_size, input_size,
                  output_size, num_module_neurons, num_tasks)

task_manager = TaskManager(pathnet, genetic_agents, data_splits,
                           num_tasks, max_num_genetic_epochs,
                           num_path_epochs, num_samples_per_epoch)

task_manager.train_models()
