#from keras.layers.merge import _Merge
from keras.layers import Layer
import keras.backend as K

class ReduceSum(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ReduceSum, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        num_inputs = len(inputs)
        if num_inputs == 1:
            return inputs
        expanded_tensors = [K.expand_dims(x, axis=-1) for x in inputs]
        concatenated_tensor = K.concatenate(expanded_tensors, axis=self.axis)
        reduced_sum_tensor = K.sum(concatenated_tensor, axis=self.axis)
        return reduced_sum_tensor

    def compute_output_shape(self, input_shape):
        if type(input_shape) == list:
            output_shape = input_shape[0]
        else:
            output_shape = input_shape
        return tuple(output_shape)

class ReduceAverage(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ReduceAverage, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        num_inputs = len(inputs)
        if num_inputs == 1:
            return inputs
        expanded_tensors = [K.expand_dims(x, axis=-1) for x in inputs]
        concatenated_tensor = K.concatenate(expanded_tensors, axis=self.axis)
        reduced_average_tensor = (K.sum(concatenated_tensor, axis=self.axis) /
                                                        float(num_inputs))
        return reduced_average_tensor

    def compute_output_shape(self, input_shape):
        if type(input_shape) == list:
            output_shape = input_shape[0]
        else:
            output_shape = input_shape
        return tuple(output_shape)








if __name__ == '__main__':
    from keras.layers import Dense
    from keras.layers import Input
    from keras.models import Model
    from keras.utils import plot_model

    # input layer
    input_layer = Input(shape=(20,))

    # first layer
    path_1 = Dense(20)(input_layer)
    path_2 = Dense(20)(input_layer)
    path_3 = Dense(20)(input_layer)
    layer_paths_1 = [path_1, path_2, path_3]
    reduced_sum_1 = ReduceSum(axis=-1)(layer_paths_1)

    # second layer
    path_4 = Dense(20)(reduced_sum_1)
    path_5 = Dense(20)(reduced_sum_1)
    path_6 = Dense(20)(reduced_sum_1)
    layer_paths_2 = [path_4, path_5, path_6]
    reduced_sum_2 = ReduceAverage(axis=-1)(layer_paths_2)

    # output layer
    classification = Dense(10)(reduced_sum_2)
    model = Model(inputs=input_layer, outputs=classification)

    plot_model(model, to_file='../images/pathnet_example.png')
