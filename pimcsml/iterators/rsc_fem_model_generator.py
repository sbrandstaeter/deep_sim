import numpy as np
from numpy.random import seed
from numpy import random as rnd
import os
import subprocess

from .iterator import Iterator

# Import cubitpy module.
from cubitpy import CubitPy, cupy, get_surface_center
from cubitpy.mesh_creation_functions import create_brick

class RoughSurfaceFemModelGenerator(Iterator): 
    
    def __init__(self, num_simulations, result_description, driver, parameters, global_settings):
        super(RoughSurfaceFemModelGenerator, self).__init__(None, global_settings)
        self.num_simulations = num_simulations
        self.result_description = result_description
        self.driver = driver
        self.parameters = parameters
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):

        print(config.get("experiment_name"))

        method_options = config["method"]["method_options"]
        num_simulations = method_options.get("num_simulations", None)
        result_description = method_options.get("result_description", None)
        
        driver = config.get("driver", None)
        parameters = config.get("parameters", None)
        global_settings = config.get("global_settings", None)

        return cls(num_simulations, result_description, driver, parameters, global_settings)

    def run_simulation(self):

        # what should be done here?
        # - read the parameters from the input file
        # - make a loop for each simulation (generation of models)
        #     - create the rough surface using RMP (random mid point approach) 
        #     - generate blocks
        #     - define BCs and header file
        #     - store the result
        
        n_global = self.parameters["geometrical_parameters"]["n"]
        H_global = self.parameters["geometrical_parameters"]["H"]

        n = n_global["distribution_parameter"]

        start = H_global["distribution_parameter"][0]
        end = H_global["distribution_parameter"][1]
        # seeds = H_global["size"] # since we will use only h as our interest set seeds as the number of simulations
        seeds = self.num_simulations

        H_iterator = np.linspace(start, end, seeds)

        for i in range(self.num_simulations):
            
            H = H_iterator[i]
            # generate rough surface using RMP
            input_path = self.generate_2D_surface(n, H, i)
            # generate blocks
            self.generate_blocks(input_path, n, i) 

    def generate_2D_surface(self,n,H,iter):
        '''
        generates rough surfaces using RMP (random mid point) approach
        '''
        N = 2**n     
        z = np.zeros([N+1,N+1])

        alpha = 1 / np.sqrt(0.09)

        D = N
        d = N//2

        for _ in range(n):
            alpha=alpha/np.sqrt(2)**H
            
            for j in range(d,N-d+1,D):
                for k in range(d,N-d+1,D):
                    z[j,k] =  (z[j+d,k+d]+z[j+d,k-d]+z[j-d,k+d]+z[j-d,k-d])/4+alpha*rnd.randn()
            
            alpha=alpha/np.sqrt(2)**H
            
            for j in range(d,N-d+1,D):
                z[j,0]=(z[j+d,0]+z[j-d,0]+z[j,d])/3+alpha*rnd.randn()
                z[j,N]=(z[j+d,N]+z[j-d,N]+z[j,N-d])/3+alpha*rnd.randn()
                z[0,j]=(z[0,j+d]+z[0,j-d]+z[d,j])/3+alpha*rnd.randn()
                z[N,j]=(z[N,j+d]+z[N,j-d]+z[N-d,j])/3+alpha*rnd.randn()
    
            for j in range(d,N-d+1,D):
                for k in range(D,N-d+1,D):
                    z[j,k]=(z[j,k+d]+z[j,k-d]+z[j+d,k]+z[j-d,k])/4+alpha*rnd.randn()
            
            for j in range(D,N-d+1,D):
                for k in range(d,N-d+1,D):
                    z[j,k]=(z[j,k+d]+z[j,k-d]+z[j+d,k]+z[j-d,k])/4+alpha*rnd.randn()


            D = D//2
            d = d//2 

        zref = self.parameters["geometrical_parameters"]["zref"].get("distribution_parameter")
        scalefactor = zref/(np.max(z)-np.mean(z))
        z = z*scalefactor
        z = z-(np.min(z))

        path_out = self.global_settings["output_dir"]
        full_path = path_out + "/rough_surface_" + str(iter) + ".dat"
        np.savetxt(full_path, z, delimiter=';')
        
        return full_path 

    def rough_block(self, cubit, z_surf, nx, ny, nz):
        """
        Create the solid mesh for the plane curved test.

        Return groups with the nodes at the bottom and top surface.
        """

        def _get_node_index(i_x, i_y, i_z):
            return (nz + 1) * (ny + 1) * i_x + (nz + 1) * i_y + i_z

        def _get_element_index(i_x, i_y, i_z):
            return nz * ny * i_x + nz * i_y + i_z

        x_range = np.linspace(0.,1.0,nx+1)
        y_range = np.linspace(0.,1.0,ny+1) 
        # Create the nodes.
        coordinates = np.zeros([(nx + 1) * (ny + 1) * (nz + 1), 3])
        for i_x in range(nx+1):
            x = x_range[i_x]
            for i_y in range(ny+1):
                y = y_range[i_y]
                z_top = z_surf[i_x*(nx+1)+i_y] + 1.0
                for i_z in range(nz+1):
                    z = i_z * z_top / nz
                    if (z < 0):
                        raise ValueError('The function should be positive in the '
                            'whole block.')
                    coordinates[_get_node_index(i_x, i_y, i_z), :] = [x, y, z[0]]

        # Create the elements.
        topology = np.zeros([nx * ny * nz, 8], dtype=int)
        for i_x in range(nx):
            for i_y in range(ny):
                for i_z in range(nz):
                    element_index = _get_element_index(i_x, i_y, i_z)
                    node_index_1 = _get_node_index(i_x, i_y, i_z)
                    node_index_2 = _get_node_index(i_x + 1, i_y, i_z)
                    node_index_3 = _get_node_index(i_x + 1, i_y + 1, i_z)
                    node_index_4 = _get_node_index(i_x, i_y + 1, i_z)
                    topology[element_index, :] = [
                        node_index_1,
                        node_index_2,
                        node_index_3,
                        node_index_4,
                        node_index_1 + 1,
                        node_index_2 + 1,
                        node_index_3 + 1,
                        node_index_4 + 1
                        ]

        # Get the start index for hex and nodes.
        i_node_base = cubit.get_node_count()
        i_element_base = cubit.get_hex_count()
        for coord in coordinates:
            coord_string = ' '.join(map(str, coord))
            cubit.cmd('create node location {}'.format(coord_string))
        for element_topology in topology:
            element_string = ' '.join(map(str, i_node_base + element_topology + 1))
            cubit.cmd('create hex node {}'.format(element_string))

        # Return the groups containing relevant geometry items.
        bottom = cubit.group()
        top = cubit.group()
        for i_x in range(nx + 1):
            for i_y in range(ny + 1):
                bottom_index = _get_node_index(i_x, i_y, 0)
                bottom.add('add node {}'.format(bottom_index + i_node_base + 1))
                top_index = _get_node_index(i_x, i_y, nz)
                top.add('add node {}'.format(top_index + i_node_base + 1))
        #all_nodes = cubit.group(name='function_block_all')
        #for i in range(len(coordinates)):
        #    all_nodes.add('add node {}'.format(i + i_node_base + 1))
        elements = cubit.group(name='Block_rough')
        for i in range(len(topology)):
            elements.add('add hex {}'.format(i + i_element_base + 1))

        cubit.add_element_type(elements, cupy.element_type.hex8,
                name=None, material='MAT 2',
                bc_description='KINEM nonlinear EAS none')

        # apply BC for the rough surface
        # dirichlet
        cubit.add_node_set(bottom, name='fix',
                        bc_section='DESIGN SURF DIRICH CONDITIONS',
                        bc_description='NUMDOF 3 ONOFF 1 1 1 '
                            + 'VAL 0.0 0.0 0.0 '
                            + 'FUNCT 0 0 0')
        
        cubit.add_node_set(top, name='slave',
                        bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                        bc_description='1 Slave')


        return {
            #'all': all_nodes,
            'top': top,
            'bottom': bottom,
            'hex': elements
            }


    def generate_blocks(self,surface_path, n, iter):
        """
        generates FEM input for the rough and flat blocks
        """
        # initialize cubit
        cubit = CubitPy()

        ## generate flat block (here it is assumed that flat block lays on the top)

        block_size = [1, 1, 1] # decide the size of the block
        n_elements = [1, 1, 1] # decide the number of element for each direction

        block_flat = create_brick(cubit, block_size[0], block_size[1], block_size[2], mesh_interval=[n_elements[0], n_elements[1], n_elements[2]],
            mesh=False) # create the block, do not mesh it

        movement = [0.5,0.5,1.7] # define the dimension for shifting the flat block, 
                                 # these values are completely temporary so they will change depending on
                                 # the result of the blocks so this must be automatized as well
        cubit.move(block_flat, movement) # move it 

        block_flat.mesh() # mesh the flat block

        
        
        # define the BC for the flat block
        # idea here:
            # define the bottom surface of the flat block as master 
            # apply dirichlet on the top surface
             
        for _, surf in enumerate(block_flat.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))                
            if np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='load_upper',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 3 ONOFF 1 1 1 '
                        + 'VAL 0.0 0.0 -0.6 '
                        + 'FUNCT 1 1 2 ')
            elif np.dot(normal, [0, 0, 1]) == -1:
                cubit.add_node_set(surf, name='master',
                    bc_section='DESIGN SURF MORTAR CONTACT CONDITIONS 3D',
                    bc_description='1 Master')

        ## generate the rough block (here it is assumed that rough block lays on the bottom)

        n_elem_rough = 2**n  # this gives the number of element on the rough surface

        # load the rough surface topography
        z_surf = np.loadtxt(fname=surface_path,delimiter=";",usecols=range(n_elem_rough+1)) 
        z_surf = z_surf.reshape((n_elem_rough+1)*(n_elem_rough+1),1)/500.0 # height of rough surfaces are calculated not in m
                                                                           # so we have to divide heights to a number, i.e. 500
                                                                           # but this is arbitrary as well 

        # create the lower rough block
        self.rough_block(cubit, z_surf, n_elem_rough, n_elem_rough, n_elem_rough)

        output_dir = self.global_settings["output_dir"]
        
        # create dat, exo and cub files
        file_name = 'rsc_fem_model' + str(iter)
        final_path_names = os.path.join(output_dir, file_name)
        
        cubit.export_cub(final_path_names + '.cub')
        cubit.export_exo(final_path_names + '.exo')

        cubit.head = self.get_contact_header() # add the header file

        cubit.create_dat(final_path_names + '.dat')

        #cubit.display_in_cubit()


    def get_contact_header(self):
        
        header = '''
        -----------------------------------------------------------------PROBLEM TYP
        PROBLEMTYP                      Structure
        RESTART                         0
        --------------------------------------------------------------DISCRETISATION
        NUMFLUIDDIS                     0
        NUMSTRUCDIS                     1
        NUMALEDIS                       0
        NUMTHERMDIS                     0
        --------------------------------------------------------------------------IO
        OUTPUT_BIN                      Yes
        STRUCT_DISP                     Yes
        STRUCT_STRESS                   No
        FILESTEPS                       1000
        ----------------------------------------------------------STRUCTURAL DYNAMIC
        LINEAR_SOLVER                   1
        DYNAMICTYP                      GenAlpha
        RESULTSEVRY                     1
        RESTARTEVRY                     1
        NLNSOL                          fullnewton
        TIMESTEP                        0.02
        NUMSTEP                         30
        MAXTIME                         0.6
        DAMPING                         Yes
        M_DAMP                          0.00001
        K_DAMP                          0.00001
        TOLRES                          1.0E-8
        TOLDISP                         1.0E-8
        NORM_RESF                       Abs
        NORM_DISP                       Abs
        NORMCOMBI_RESFDISP              And
        MAXITER                         50
        INT_STRATEGY                    Standard
        -------------------------------------------------STRUCTURAL DYNAMIC/GENALPHA
        GENAVG                          TrLike
        RHO_INF                         1.0
        -------------------------------------------------------------CONTACT DYNAMIC
        LINEAR_SOLVER                   2
        STRATEGY                        Penalty
        PENALTYPARAM                    20000
        FRICTION                        None
        SEMI_SMOOTH_NEWTON              Yes
        SEMI_SMOOTH_CN                  1.0
        -------------------------------------------------------------MORTAR COUPLING
        LM_SHAPEFCN                     Standard
        SEARCH_ALGORITHM                BruteForceEleBased
        SEARCH_PARAM                    0.3
        LM_DUAL_CONSISTENT              none
        --------------------------------------------------------------------SOLVER 1
        NAME                            Structure_Solver
        SOLVER                          UMFPACK
        --------------------------------------------------------------------SOLVER 2
        NAME                            Contact_Solver
        SOLVER                          UMFPACK
        -------------------------------------------------------------------MATERIALS
        //                              MAT_Struct_StVenantKirchhoff
        MAT 1 MAT_ElastHyper NUMMAT 1 MATIDS 3 DENS 7.8e-6
        MAT 3 ELAST_CoupNeoHooke YOUNG 210.0 NUE 0.3
        MAT 2 MAT_ElastHyper NUMMAT 1 MATIDS 4 DENS 7.8e-6
        MAT 4 ELAST_CoupNeoHooke YOUNG 2100.0 NUE 0.3
        ----------------------------------------------------------------------FUNCT1
        COMPONENT 0 FUNCTION a
        VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 4 TIMES 0 0.5 0.6 1 VALUES 0 0 0.2 0.2
        ----------------------------------------------------------------------FUNCT2
        COMPONENT 0 FUNCTION a
        VARIABLE 0 NAME a TYPE linearinterpolation NUMPOINTS 3 TIMES 0 0.5 1 VALUES 0 1 1
        '''

        return header