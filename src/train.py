from data_manager import DataManager
from genetic_agents import GeneticAgents
from pathnet import PathNet
from utils import normalize_images
from utils import split_data
from utils import load_layer_weights
from utils import save_layer_weights
from utils import to_categorical

# parameters 
num_layers = 5
num_modules_per_layer = 10
num_genetic_epochs = 100
num_cnn_epochs = 100
batch_size = 32
num_genotypes_per_tournament = 2
validation_split = .2
verbosity = 1
save_path = '../trained_models/'
mnist_args = [6 , 7]
num_classes = len(mnist_args)

# instantiating agents and models
genetic_agents = GeneticAgents(shape=(num_modules_per_layer, num_layers))
pathnet = PathNet(shape=(num_modules_per_layer, num_layers))

# loading data splits
data_manager = DataManager()
train_data, test_data = data_manager.load('mnist', class_args=mnist_args)
train_data, validation_data = split_data(train_data)
train_images, train_classes = train_data
validation_images, validation_classes = validation_data
test_images, test_classes = test_data

# encode outputs into one hot vectors
class_arg_splits = (train_classes, validation_classes, test_classes)
categorical_data = [to_categorical(class_arg_split)
                    for class_arg_split in class_arg_splits]
train_classes, validation_classes, test_classes = categorical_data

# normalize images
image_splits = (train_images, validation_images, test_images)
normalized_images = [normalize_images(image_split)
                    for image_split in image_splits]
train_images, validation_images, test_images = normalized_images

# train paths with an evolution strategy
for genetic_epoch_arg in range(num_genetic_epochs):
    sampled_paths, sampled_args  = genetic_agents.sample_genotype_paths(
                                            num_genotypes_per_tournament)
    losses = []
    for genotype_path in sampled_paths:
        path_model = pathnet.compile_path(genotype_path)
        load_layer_weights(path_model, save_path)
        path_model.compile(optimizer='adam', loss='categorical_crossentropy',
                                                        metrics=['accuracy'])
        path_model.fit(train_images, train_classes, batch_size, num_cnn_epochs,
                                    verbosity, validation_data=validation_data,
                                                                  shuffle=True)
        save_layer_weights(path_model, save_path)
        score = path_model.evaluate(test_images, test_classes)
        losses.append(-1 * score[0])
    genetic_agents.overwrite(sampled_args, losses)
