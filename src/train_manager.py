from keras.optimizers import SGD
from data_manager import DataManager
from utils import load_layer_weights
from utils import save_layer_weights
from utils import shuffle
from utils import reset_weights
from utils import from_path_to_names
from utils import flatten


class TaskManager(object):
    def __init__(self, pathnet, genetic_agents, data_splits, num_tasks,
                 max_num_genetic_epochs=500,
                 num_path_epochs=100,
                 num_samples_per_path=16*50,
                 batch_size=16,
                 optimizer=SGD(lr=0.0001),
                 accuracy_treshold=.998,
                 save_path='../trained_models/'):

        self.pathnet = pathnet
        self.genetic_agents = genetic_agents
        self.data_splits = data_splits
        self.num_tasks = num_tasks
        self.optimizer = optimizer
        self.save_path = save_path
        self.max_num_genetic_epochs = max_num_genetic_epochs
        self.batch_size = batch_size
        self.num_path_epochs = num_path_epochs
        self.verbosity = 0
        self.num_samples_per_path = num_samples_per_path
        self.accuracy_treshold = accuracy_treshold
        self.frozen_paths = []
        self.genetic_epochs = []

    def _get_model_data(self, model_arg):
        train_data, val_data, test_data = self.data_splits
        train_features = train_data[0][model_arg]
        train_labels = train_data[1][model_arg]
        val_features = val_data[0][model_arg]
        val_labels = val_data[1][model_arg]
        test_features = test_data[0][model_arg]
        test_labels = test_data[1][model_arg]
        model_train_data = (train_features, train_labels)
        model_val_data = (val_features, val_labels)
        model_test_data = (test_features, test_labels)
        return (model_train_data, model_val_data, model_test_data)

    def train_models(self):
        # train paths with an evolution strategy
        for task_arg in range(self.num_tasks):
            task_data_splits = self._get_model_data(task_arg)
            train_data, validation_data, test_data = task_data_splits
            train_images, train_classes = train_data
            validation_images, validation_classes = validation_data
            for genetic_epoch_arg in range(self.max_num_genetic_epochs):
                print('*'*30)
                print('Genetic epoch:', genetic_epoch_arg)
                sampled_paths, sampled_args = (
                        self.genetic_agents.sample_genotype_paths(2))
                train_images, train_classes = shuffle(train_images,
                                                      train_classes)
                sampled_train_images = train_images[
                        :self.num_samples_per_path]
                sampled_train_classes = train_classes[
                        :self.num_samples_per_path]
                fitness_values = []
                for genotype_path in sampled_paths:
                    path_model = self.pathnet.build(genotype_path, task_arg)
                    load_layer_weights(path_model, self.save_path)
                    path_model.compile(
                            optimizer=self.optimizer,
                            loss='mean_squared_error', metrics=['acc'])
                    frozen_path_names = from_path_to_names(self.frozen_paths)
                    for path_layer in path_model.layers:
                        if path_layer.name in frozen_path_names:
                            path_layer.trainable = False

                    sampled_train_images = flatten(sampled_train_images)
                    sampled_train_classes = flatten(sampled_train_classes)
                    path_model.fit(
                            sampled_train_images, sampled_train_classes,
                            self.batch_size, self.num_path_epochs,
                            self.verbosity, shuffle=True)
                    save_layer_weights(
                            path_model, self.save_path, self.frozen_paths)

                    # fix this evaluation
                    train_images_hack = flatten(train_images)
                    train_classes_hack = flatten(train_classes)
                    score = path_model.evaluate(
                            train_images_hack, train_classes_hack,
                            verbose=self.verbosity)
                    loss, accuracy = score
                    print('Loss: %.3f Accuracy: %.3f' % (loss, accuracy))
                    fitness_values.append(-1 * loss)
                best_path = self.genetic_agents.overwrite(
                        sampled_args, fitness_values)
                if accuracy > self.accuracy_treshold:
                    self.frozen_paths.append(best_path)
                    print('Frozen paths: \n', self.frozen_paths[task_arg])
                    reset_weights(self.frozen_paths, self.save_path)
                    self.genetic_epochs.append(genetic_epoch_arg)
                    break


if __name__ == "__main__":
    from genetic_agents import GeneticAgents
    from pathnet import PathNet

    # parameters for genetic agents and path net architecture
    num_layers = 3
    num_modules_per_layer = 10
    num_neurons_per_module = 20
    output_size = 2
    genetic_agents = GeneticAgents(shape=(num_modules_per_layer, num_layers))
    pathnet = PathNet(shape=(num_modules_per_layer, num_layers),
                    num_neurons_per_module=num_neurons_per_module,
                    output_size=output_size)

    # parameters for the task manager
    class_arg_list = [[4, 5], [6, 7]]
    accuracy_treshold = .90
    dataset_name = 'mnist'
    task_manager = TaskManager(pathnet, genetic_agents,
                            dataset_name, class_arg_list,
                            accuracy_treshold=accuracy_treshold)
    task_manager.train_tasks()
