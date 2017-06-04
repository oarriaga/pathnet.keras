from keras.utils import to_categorical

from data_manager import DataManager
from genetic_path import GeneticPath
from pathnet import PathNet
from utils import normalize_images
from utils import split_data

# parameters 
num_layers = 5
num_modules_per_layer = 10
num_genetic_epochs = 100
num_cnn_epochs = 100
batch_size = 32
num_genotypes_per_tournament = 2
validation_split = .2
verbosity = 1

# instantiating classes and process data
genetic_paths = GeneticPath(shape=(num_modules_per_layer, num_layers))
pathnet = PathNet(shape=(num_modules_per_layer, num_layers))

# loading and pre-processing dataset
data_manager = DataManager()
train_data, test_data = data_manager.load('mnist', class_args=[6, 7])
train_data, validation_data = split_data(train_data)
train_images, train_classes = train_data
validation_images, validation_classes = validation_data
test_images, test_classes = test_data

arg_classes = (train_classes, validation_classes, test_classes)
one_hot_classes = [to_categorical(class_data) for class_data in arg_classes]
train_classes, validation_classes, test_classes = one_hot_classes

image_data = (train_images, validation_images, test_images)
normalized_images = [normalize_images(image) for image in image_data]
train_images, validation_images, test_images = normalized_images

# train path with evolution strategies
for genetic_epoch_arg in range(num_genetic_epochs):
    sampled_paths, sampled_args  = genetic_paths.sample_genotype_paths(
                                            num_genotypes_per_tournament)
    losses = []
    for genotype_path in sampled_paths:
        path_model = pathnet.compile_path(genotype_path)
        path_model.compile(optimizer='adam', loss='categorical_crossentropy',
                                                        metrics=['accuracy'])
        path_model.fit(train_images, train_classes, batch_size, num_cnn_epochs,
                                    verbosity, validation_data=validation_data,
                                                                  shuffle=True)
        score = path_model.evaluate(test_images, test_classes)
        losses.append(-1 * score[0])
    genetic_paths.overwrite(sampled_args, losses)


