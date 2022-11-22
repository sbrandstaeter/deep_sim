import abc

class Iterator (metaclass=abc.ABCMeta):
    
    def __init__(self, model=None, global_settings=None):
        self.global_settings = global_settings

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):
        '''
        Creates iterator from problem description

        Parameters
        ----------
        config : dict
            contains the input parameters
        iterator_name : str , optional
            iterator to be run to distinguish the correct iterator

        Returns
        -------
        cls(num_simulations, result_description, parameters, sampling, global_settings)
            a class object of the chosen iterator
        '''
        
        if config['method']['method_name'] == "rsc_bem_rmd":
            from .rsc_bem_rmd_iterator import RoughSurfaceBemRMDIterator as main_iter
        elif config['method']['method_name'] == "rsc_fem_model":
            from .rsc_fem_model_generator import RoughSurfaceFemModelGenerator as main_iter
        elif config['method']['method_name'] == "baci_fem":
            from .baci_fem_iterator import BaciFemIterator as main_iter
        elif config['method']['method_name'] == "ml_trainer":
            from .ml_iterator import MachineLearningIterator as main_iter
        elif config['method']['method_name'] == "beam_fem_model":
            from .beam_fem_model_generator import BeamFemModelGenerator as main_iter
        else:
            raise Exception("Method does not exits!")
        
        method_dict = {config['method']['method_name']  : main_iter}
        
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