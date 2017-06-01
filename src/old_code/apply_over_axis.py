import numpy as np
def generate_valid_module():
    num_modules_per_layer = 10
    max_num_active_paths = 3
    module = np.zeros(num_modules_per_layer)
    random_args = np.random.permutation(num_modules_per_layer)
    num_active_paths = np.random.randint(1, max_num_active_paths + 1)
    random_args = np.unravel_index(random_args[:num_active_paths],
                                    dims=num_modules_per_layer)
    module[random_args] = 1

def generate_valid_module2(module):
    max_num_active_paths = 3
    num_modules_per_layer = len(module)
    random_args = np.random.permutation(num_modules_per_layer)
    num_active_paths = np.random.randint(1, max_num_active_paths + 1)
    random_args = np.unravel_index(random_args[:num_active_paths],
                                    dims=num_modules_per_layer)
    module[random_args] = 1
    return module

path = np.zeros(shape=(5,10))
path = np.apply_along_axis(generate_valid_module2, axis=0, arr=path)
print(path)



