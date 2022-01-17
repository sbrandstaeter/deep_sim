import numpy as np
from numpy.random import seed
import os
import pandas as pd
import collections

from .iterator import Iterator
from ..sampling.sampling import Sampling

from cubitpy import CubitPy, cupy, get_surface_center
from cubitpy.mesh_creation_functions import create_brick

class BeamFemModelGenerator(Iterator):
    '''
    A class to generate multiply input files of a beam structure to run FEM simulations for BACI in an automated way

    Attributes
    ----------
    num_simulations : int
        total number of simulations to be run
    result_description : dict
        properties of the result file
    parameters : dict
        parameters of the problem (geometrical or material)
    sampling : dict
        properties of the chosen sampling strategy
    global_settings :  dict
        contains output directory and experiment name independent of the given problem
    ''' 

    def __init__(self, num_simulations, result_description, parameters, sampling, global_settings):
        super(BeamFemModelGenerator, self).__init__(None, global_settings)
        self.num_simulations = num_simulations
        self.result_description = result_description
        self.parameters = parameters
        self.sampling = sampling
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):
        '''
        Creates the iterator from problem description

        Parameters
        ----------
        config : dict
            contains the input parameters
        iterator_name : str , optional
            iterator to be run

        Returns
        -------
        cls(num_simulations, result_description, parameters, sampling, global_settings)
            a class object of the chosen iterator
        '''

        print(config.get("global_settings", None)["experiment_name"])

        method_options = config["method"]["method_options"]
        num_simulations = method_options.get("num_simulations", None)
        result_description = method_options.get("result_description", None)
        
        parameters = config.get("parameters", None)
        sampling = config.get("sampling", None)
        global_settings = config.get("global_settings", None)

        return cls(num_simulations, result_description, parameters, sampling, global_settings)

    def run_simulation(self):
        '''
        Generate the input files
        '''       
        # sampling methods require the ranges of the parameters
        # first get the parameters from the input file
        domain = self.get_parameters()

        # create sampling instance
        sample_model = Sampling(domain=list(domain.values()), 
                        n_samples=self.num_simulations, 
                        sampling_name=self.sampling["sampling_name"], 
                        sampling_options=self.sampling["sampling_options"])

        # generate the samples
        sampled_parameters = np.array(sample_model.generate_samples())
        
        # match the sampled parameters with the domain, 
        # so we are sure that correct parameter names are matched the corresponding value
        j = 0
        for param_name, _ in domain.items():
            if self.parameters["material_parameters"].get(param_name):
                domain[param_name] = sampled_parameters[:,j].astype(self.parameters["material_parameters"][param_name].get("type"))
            else:
                domain[param_name] = sampled_parameters[:,j].astype(self.parameters["geometrical_parameters"][param_name].get("type"))
            j += 1


        # create a dictionary to store 
        simulation_dict={"simulation_name":[]}
        
        # iterate through the simulations
        for i in range(self.num_simulations):
            simulation_file_name = self.beam_3d(domain=domain, current_sim=i)
            simulation_dict['simulation_name'].append(simulation_file_name)

        # write results into a file
        if(self.result_description.get("write_results")):
            
            result_file_name = self.result_description["result_file_name"] + "." + self.result_description["result_file_format"]
            
            df = pd.DataFrame(domain)
            df["simulation_name"] = simulation_dict["simulation_name"]

            output_dir = os.path.join(self.global_settings['output_dir'], result_file_name)

            df.to_csv(output_dir)    

    def beam_3d(self, domain, current_sim):
        """
        Creates the beam structure using CubitPY
        
        Parameters
        ----------
            domain : dict
                contains the parameter of the problem
            current_sim : int
                current simulation number

        Returns
        -------
            simulation_file_name : str
                Baci input file 
        """

        # initialize the cubit
        cubit = CubitPy()

        # get the dimensions from the input file
        L = domain["L"][current_sim]
        t = domain["t"][current_sim]
        k = domain["k"][current_sim]
        x = domain["x"][current_sim]
        q = domain["q"][current_sim]

        # get the material properties from the input file
        E = domain["E"][current_sim]
        nu = domain["nu"][current_sim]

        beam_size = [L, t, k]
        # beam_n_elements = [40, 4, 4] # the number of elements on each edge

        lx_1, ly_1, lz_1 = beam_size
        #nx_1, ny_1, nz_1 = beam_n_elements
        #a_n=nx_1/lx_1*a
    
        # create the brick element
        block_1 = cubit.brick(lx_1, ly_1, lz_1) 

        # create a curve on location x where the force will be applied
        start_vertices = [x-L/2,-t/2,k/2]
        end_vertices = [x-L/2,t/2,k/2]
        cubit.cmd(f'create curve location {start_vertices[0]} {start_vertices[1]} {start_vertices[2]} location {end_vertices[0]} {end_vertices[1]} {end_vertices[2]}')

        # make the partition so the curve is part of the brick
        for surf_index, surf in enumerate(block_1.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [0, 0, 1]) == 1:
                surface_id = surf_index + 1 
                cubit.cmd(f'partition create surface {surface_id} curve 13')
        
        # mesh the brick
        cubit.cmd('mesh volume 1')
        #cubit.cmd('volume 1 size auto factor 6')
        
        # assign the element type
        cubit.add_element_type(block_1.volumes()[0], cupy.element_type.hex27, #hex8 and hex27
            name='block_1', material='MAT 1',
            bc_description='KINEM nonlinear') # add EAS none for linear elements --> KINEM nonlinear EAS none

        # Create node and curve sets and apply the BCs
        
        # Nodes where BC are applied (right handside of the block) 
        flag_surface = True
        upper_curves = []
        for _, surf in enumerate(block_1.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [1, 0, 0]) == 1:
                cubit.add_node_set(surf, name='fix_right',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 1 1 1 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')

        # Nodes where BC are applied (left handside of the block)    
            elif np.dot(normal, [1, 0, 0]) == -1:
                cubit.add_node_set(surf, name='fix_left',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 1 1 1 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')
        
        # Possible curves whose one of them will be chosen as the application of the load vector  
            elif np.dot(normal, [0, 0, 1]) == 1 and flag_surface==True:
                upper_curves.append(surf.curves()) 

        # Apply the force 
        for left_curve in upper_curves[0]:
            for right_curve in upper_curves[1]:
                if left_curve.get_node_ids() == right_curve.get_node_ids():
                    #n_applied_q = len(left_curve.get_node_ids())*2-1
                    q_applied = q/t
                    cubit.add_node_set(left_curve, name='load_upper',
                                bc_section='DESIGN LINE NEUMANN CONDITIONS',
                                bc_description='NUMDOF 6 ONOFF 0 0 1 0 0 0 '
                                    + f'VAL 0.0 0.0 {q_applied} 0. 0. 0. '
                                    + 'FUNCT 0 0 0 0 0 0')

        output_dir = self.global_settings['output_dir']
        # create dat, exo and cub files
        simulation_file_name = "_".join(self.global_settings["experiment_name"].split()) + '_' + str(current_sim+1) + ".dat"
        final_path_names = os.path.join(output_dir, simulation_file_name)
        
        # cubit.export_cub(final_path_names + '.cub')
        # cubit.export_exo(final_path_names + '.exo')

        # add the header file
        cubit.head = self.get_contact_header(E, nu)

        # create the dat file 
        cubit.create_dat(final_path_names)
        
        #cubit.display_in_cubit()

        return simulation_file_name


    def get_contact_header(self, E, nu):
        '''
        Creates the header file which contains simulation parameters

        Parameters
        ----------
            E : float
                Young's modulus
            nu : float 
                Poisson's ratio

        Returns
        -------
            header : str
                Header file
        '''
        
        header = f'''
        -----------------------------------------------------DISCRETISATION
        NUMFLUIDDIS                     0
        NUMSTRUCDIS                     1
        NUMALEDIS                       0
        NUMARTNETDIS                    0
        NUMTHERMDIS                     0
        NUMAIRWAYSDIS                   0
        --------------------------------------------------------PROBLEM TYP
        PROBLEMTYP                      Structure
        //SHAPEFCT                        Polynomial
        RESTART                         0
        --------------------------------------------------------------------FUNCT1
        COMPONENT 0 FUNCTION a
        VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 2 TIMES 0 1 VALUES 0 0.1
        --------------------------------------------------------------------SOLVER 1
        NAME                            Structure_Solver
        SOLVER                          UMFPACK
        --------------------------------------------------------------------SOLVER 3
        NAME                            Structure_Solver
        SOLVER                          Aztec_MSR
        AZSOLVE                         GMRES 
        AZTOL                           1.0e-5
        AZCONV                          AZ_r0 
        AZOUTPUT                        10    
        AZPREC                          ILU   
        IFPACKGFILL                     0 //0 or 3
        --------------------------------------------------------------------SOLVER 4
        NAME                            Structure_Solver
        SOLVER                          Aztec_MSR
        AZSOLVE                         GMRES
        AZTOL                           1.0e-5
        AZCONV                          AZ_r0
        AZOUTPUT                        10
        AZPREC                          ML
        ML_MAXCOARSESIZE                8000
        ML_MAXLEVEL                     5
        ML_AGG_SIZE                     27
        ML_DAMPFINE                     1.0
        ML_DAMPMED                      1.0
        ML_DAMPCOARSE                   1.0
        ML_PROLONG_SMO                  1.33333333
        ML_SMOTIMES                     6 12 12 12 1
        ML_SMOOTHERFINE                 Chebychev
        ML_SMOOTHERMED                  Chebychev
        ML_SMOOTHERCOARSE               Superlu
        -------------------------------------------------STRUCTURAL DYNAMIC
        LINEAR_SOLVER                   4
        DYNAMICTYP                      Statics
        RESULTSEVRY                     1
        RESTARTEVRY                     20
        NLNSOL                          fullnewton                    
        TIMESTEP                        0.1
        NUMSTEP                         10
        MAXTIME                         1.0
        TOLRES                          1.0E-6
        TOLDISP                         1.0E-7
        NORM_RESF                       Abs
        NORM_DISP                       Abs
        NORMCOMBI_RESFDISP              And
        MAXITER                         50
        DIVERCONT                       stop

        --------------------------------------------------------------------------IO
        OUTPUT_BIN                      Yes
        STRUCT_DISP                     Yes
        STRUCT_STRESS                   Yes
        -------------------------------------------------------------------MATERIALS
        MAT 1 MAT_Struct_StVenantKirchhoff YOUNG {E} NUE {nu} DENS 0.0
        '''

        return header

    def get_parameters(self):
        '''
        Obtains the parameters from the input file and generates the range for each parameter

            Returns
            -------
                domain : OrderedDict
                    contains the range of the parameters of the problem

            Example
            -------
            Let's chech how `domain` looks like

            >>> print(domain)
            OrderedDict([('E', array([ 50000.   ,  ...0000.   ])), ('nu', array([0.2  , 0.232,...   0.49 ])), ('L', [40]), ('t', array([1.   , 1.222,...   3.   ])), ('k', array([1.   , 1.222,...   3.   ])), ('x', array([ 1.   ,  5.22..., 39.   ])), ('q', array([ -10.,  -20.,...   -100.]))])

        '''
    
        mat_params = self.parameters["material_parameters"]
        geo_params = self.parameters["geometrical_parameters"]

        domain = collections.OrderedDict()

        for param, value in mat_params.items():
            if value["size"] != 1:
                domain[param] = np.linspace(value["distribution_parameter"][0],value["distribution_parameter"][1],value["size"]).round(3).astype(value["type"])
            else:
                domain[param] = [value["distribution_parameter"]]
    
        for param, value in geo_params.items():
            if value["size"] != 1:
                domain[param] = np.linspace(value["distribution_parameter"][0],value["distribution_parameter"][1],value["size"]).round(3).astype(value["type"])
            else:
                domain[param] = [value["distribution_parameter"]]
    
        return domain