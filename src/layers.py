from keras.layers.merge import _Merge
import keras.backend as K

class ReduceSum(_Merge):
    """Layer the reduce sum module from path-nets.

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
        expanded_tensors = [K.expand_dims(x, axis=-1) for x in inputs]
        concatenated_tensor = K.concatenate(expanded_tensors, axis=self.axis)
        reduced_sum_tensor = K.sum(concatenated_tensor, axis=self.axis)
        return reduced_sum_tensor
        #return K.concatenate(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `ReduceSum` layer should be called '
                             'on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        """
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        """
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
        base_config = super(ReduceSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



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
    reduced_sum_2 = ReduceSum(axis=-1)(layer_paths_2)

    # output layer
    classification = Dense(10)(reduced_sum_2)
    model = Model(inputs=input_layer, outputs=classification)

    plot_model(model, to_file='../images/pathnet_example.png')
