from keras.layers.merge import _Merge
from keras.layers import Dense
from keras.layers import Input
from keras.layers import concatenate
from keras.models import Model
from keras.layers import Lambda
import keras.backend as K
from keras.utils import plot_model

class ReduceSum(_Merge):
    """Layer that concatenates a list of inputs.

    It takes as input a list of tensors,
    all of the same shape expect for the concatenation axis,
    and returns a single tensor, the concatenation of all inputs.

    # Arguments
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, axis=-1, **kwargs):
        super(ReduceSum, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`ReduceSum` layer should be called '
                             'on a list of inputs')
        if all([shape is None for shape in input_shape]):
            return
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('`ReduceSum` layer requires '
                             'inputs with matching shapes '
                             'except for the concat axis. '
                             'Got inputs shapes: %s' % (input_shape))

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `ReduceSum` layer should be called '
                             'on a list of inputs.')

        expanded_tensors = [K.expand_dims(x) for x in self.inputs]
        concatenated_tensor = K.concatenate(expanded_tensors, axis=self.axis)
        return K.sum(concatenated_tensor, axis=self.axis)
        #return K.concatenate(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        # Make a list of masks while making sure
        # the dimensionality of each mask
        # is the same as the corresponding input.
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                # Input is unmasked. Append all 1s to masks,
                # but cast it to bool first
                masks.append(K.cast(K.ones_like(input_i), 'bool'))
            elif K.ndim(mask_i) < K.ndim(input_i):
                # Mask is smaller than the input, expand it
                masks.append(K.expand_dims(mask_i))
            else:
                masks.append(mask_i)
        concatenated = K.concatenate(masks, axis=self.axis)
        return K.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(Concatenate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))








def ExpandDims(x):
    return K.expand_dims(x, axis=-1)

def ReduceSum(x):
    return K.sum(x, axis=-1)

"""
def ReduceSumModule(tensors):
    expanded_tensors = [K.expand_dims(x, axis=-1) for x in tensors]
    concatenated_tensors = concatenate(expanded_tensors, axis=-1)
    return K.sum(concatenated_tensors, axis=-1)
"""

input_layer = Input(shape=(20,))

path_1 = Dense(20)(input_layer)
path_2 = Dense(20)(input_layer)
path_3 = Dense(20)(input_layer)

paths = [path_1, path_2, path_3]
concatenated_paths_1 = concatenate(paths, axis=-1)
reduced_sum_1 = Lambda(ReduceSum)(concatenated_paths_1)

path_4 = Dense(20)(reduced_sum_1)
path_4 = Lambda(ExpandDims)(path_4)

path_5 = Dense(20)(reduced_sum_1)
path_5 = Lambda(ExpandDims)(path_5)

path_6 = Dense(20)(reduced_sum_1)
path_6 = Lambda(ExpandDims)(path_6)

paths = [path_4, path_5, path_6]
concatenated_paths_2 = concatenate(paths, axis=-1)
reduced_sum_2 = Lambda(ReduceSum)(concatenated_paths_2)

classification = Dense(10)(reduced_sum_2)
model = Model(inputs=input_layer, outputs=classification)

plot_model(model, to_file='pathnet.png')
"

"""
input_layer = Input(shape=(20,))

path_1 = Dense(20)(input_layer)
path_1 = Lambda(ExpandDims)(path_1)

path_2 = Dense(20)(input_layer)
path_2 = Lambda(ExpandDims)(path_2)

path_3 = Dense(20)(input_layer)
path_3 = Lambda(ExpandDims)(path_3)

paths = [path_1, path_2, path_3]
concatenated_paths_1 = concatenate(paths, axis=-1)
reduced_sum_1 = Lambda(ReduceSum)(concatenated_paths_1)

path_4 = Dense(20)(reduced_sum_1)
path_4 = Lambda(ExpandDims)(path_4)

path_5 = Dense(20)(reduced_sum_1)
path_5 = Lambda(ExpandDims)(path_5)

path_6 = Dense(20)(reduced_sum_1)
path_6 = Lambda(ExpandDims)(path_6)

paths = [path_4, path_5, path_6]
concatenated_paths_2 = concatenate(paths, axis=-1)
reduced_sum_2 = Lambda(ReduceSum)(concatenated_paths_2)

classification = Dense(10)(reduced_sum_2)
model = Model(inputs=input_layer, outputs=classification)

plot_model(model, to_file='pathnet.png')
"""
