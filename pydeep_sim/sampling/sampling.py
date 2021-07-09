from skopt.space import Space
from skopt.sampler import Sobol
from skopt.sampler import Lhs
from skopt.sampler import Halton
from skopt.sampler import Hammersly

class Sampling:
    '''
    This class creates the samplings of the simulations

    Args
        ---
        domain : dict or tuple
            gives the ranges of the parameters to sample
        n_samples : int
            the number of samples or the number of simulations
        sampling_name : str
            the sampling method type
        sampling options :
            the options of the sampling method
    '''
    def __init__(self, domain, n_samples, sampling_name, sampling_options=None):
        self.domain = domain
        self.n_samples = n_samples
        self.sampling_name = sampling_name
        self.sampling_options = sampling_options
    
    def build_sampler(self):
        '''
        builds the sampling method from the input file 
        '''
        sampling_dict = {
            "sobol" : Sobol(),
            "lhs" : Lhs(),
            "halton" : Halton(),
            "hammersly" : Hammersly(),
            "random_sampling" : "random_sampling" 
        }

        try:
            current_sampler = sampling_dict[self.sampling_name]
        except:
            raise NameError("The chosen sampling method is not available!")

        return current_sampler

    def generate_samples(self):
        '''
        generates the samples
        '''
        space = Space(self.domain)

        if self.sampling_name == "random_sampling":
            sampled_parameters = space.rvs(self.n_samples)
        else:
            sampler = self.build_sampler()

            if self.sampling_options is not None:
                for key, value in self.sampling_options.items():
                    try:
                        sampler.__dict__[key]
                        sampler.set_params(**{key:value})
                    except:
                        raise NameError(f"The parameter {key} is not one of the correct parameter names.")
            
            sampled_parameters= sampler.generate(space.dimensions, self.n_samples)
        
        return sampled_parameters


