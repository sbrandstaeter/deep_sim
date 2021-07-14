import abc

class Iterator (metaclass=abc.ABCMeta):
    
    def __init__(self, model=None, global_settings=None):
        self.global_settings = global_settings

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):
        """ Create iterator from problem description

        Args:
            config (dict):       Dictionary with Deep_Sim problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)

        Returns:
            iterator: Iterator object

        """
        
        from .rsc_bem_rmd_iterator import RoughSurfaceBemRMDIterator
        from .rsc_fem_model_generator import RoughSurfaceFemModelGenerator
        from .baci_fem_iterator import BaciFemIterator
        from .ml_iterator import MachineLearningIterator

        method_dict = {
            'rsc_bem_rmd': RoughSurfaceBemRMDIterator,
            'rsc_fem_model': RoughSurfaceFemModelGenerator,
            'baci_fem': BaciFemIterator,
            'ml_trainer': MachineLearningIterator
        }

        if iterator_name is None:
            method_name = config['method']['method_name']
            iterator_class = method_dict[method_name]
            iterator = iterator_class.from_config_create_iterator(config)
        else:
            method_name = config[iterator_name]['method_name']
            iterator_class = method_dict[method_name]
            iterator = iterator_class.from_config_create_iterator(config, iterator_name)
        return iterator

    def run_simulation(self):
        pass

    def run(self):
        self.run_simulation()