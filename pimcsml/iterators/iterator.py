import abc

class Iterator (metaclass=abc.ABCMeta):
    
    def __init__(self, model=None, global_settings=None):
        self.global_settings = global_settings

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):
        """ Create iterator from problem description

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)

        Returns:
            iterator: Iterator object

        """
        
        from .rs_bem_iterator import RoughSurfaceBemIterator


        method_dict = {
            'rsc_bem': RoughSurfaceBemIterator,
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