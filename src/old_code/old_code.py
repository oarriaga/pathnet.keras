def _calculate_all_paths(self):
        module_paths = []
        for num_active_paths in range(self.num_active_paths):
            active_path_combinations = list(itertools.combinations(
                                        range(self.num_modules_per_layer),
                                        num_active_paths))
            module_paths = module_paths + active_path_combinations
        return module_paths
