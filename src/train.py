from keras.optimizers import SGD
from data_manager import DataManager
from genetic_agents import GeneticAgents
from pathnet import PathNet
from utils import normalize_images
from utils import split_data
from utils import load_layer_weights
from utils import save_layer_weights
from utils import to_categorical
from utils import flatten
from utils import shuffle
from utils import spice_up_images
from utils import display_image

# parameters 
num_layers = 3
num_modules_per_layer = 10
num_genetic_epochs = 100
num_cnn_epochs = 100
batch_size = 16
num_samples_per_path = 50*batch_size
num_genotypes_per_tournament = 2
num_neurons_per_module = 20
validation_split = .2
verbosity = 1
save_path = '../trained_models/'
mnist_args = [5 , 7]
num_classes = len(mnist_args)


# instantiating agents and path-net sub-models
genetic_agents = GeneticAgents(shape=(num_modules_per_layer, num_layers))
pathnet = PathNet(shape=(num_modules_per_layer, num_layers),
                num_neurons_per_module=num_neurons_per_module,
                                output_size=len(mnist_args))


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

# normalize and add salt/pepper noise to images
image_splits = (train_images, validation_images, test_images)
image_splits = [flatten(image_split) for image_split in image_splits]

image_splits = [spice_up_images(image_split)
                for image_split in image_splits]

normalized_images = [normalize_images(image_split)
                    for image_split in image_splits]

train_images, validation_images, test_images = normalized_images
validation_data = (validation_images, validation_classes)

#instantiating optimizer
sgd = SGD(lr=0.05)

# train paths with an evolution strategy
chosen_paths = []
for genetic_epoch_arg in range(num_genetic_epochs):
    sampled_paths, sampled_args  = genetic_agents.sample_genotype_paths(
                                            num_genotypes_per_tournament)
    train_images, train_classes = shuffle(train_images, train_classes)
    sampled_train_images = train_images[:num_samples_per_path]
    sampled_train_classes = train_classes[:num_samples_per_path]
    #for arg in range(len(sampled_train_images)):
        #display_image(sampled_train_images[arg].reshape(28, 28), sampled_train_classes[arg], cmap='gray')
    losses = []
    for genotype_path in sampled_paths:
        path_model = pathnet.build(genotype_path)
        load_layer_weights(path_model, save_path)
        path_model.compile(optimizer=sgd, loss='categorical_crossentropy',
                                                            metrics=['acc'])
        path_model.fit(sampled_train_images, sampled_train_classes,
                                        batch_size, num_cnn_epochs,
                                            verbosity, shuffle=True)
        save_layer_weights(path_model, save_path)
        score = path_model.evaluate(validation_images, validation_classes)
        losses.append(-1 * score[0])
    best_path = genetic_agents.overwrite(sampled_args, losses)
    chosen_paths.append(best_path)
