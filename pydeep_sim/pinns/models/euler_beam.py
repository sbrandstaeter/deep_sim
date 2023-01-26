from ..pinns import Pinns

class EulerBeam(Pinns):
    def __init__(self,num_simulations, result_description, driver, parameters, sampling, global_settings, parameter_test=None):
        super(EulerBeam, self).__init__(num_simulations, result_description, driver, parameters, sampling, global_settings)
        self.parameter_test = parameter_test
    
    def generate_input_file(self):
        '''
        Generates pyhsics for the Euler beam

        Returns
        -------
        input_dim : int
            input dimension of the problem
        output_dim : int
            output dimension of the problem
        data: DeepXDE data structure
            contains physics of the problem
        '''
        
        pressure = self.domain.get("pressure")[self.simulation_number]
        young_modulus = self.domain.get("young_modulus")[self.simulation_number]
        inertia = self.domain.get("inertia")[self.simulation_number]
        start = self.domain.get("start")[self.simulation_number]
        end = self.domain.get("end")[self.simulation_number]
        n_domain = self.domain.get("n_domain")[self.simulation_number]
        n_boundary = self.domain.get("n_boundary")[self.simulation_number]
        train_distribution = self.domain.get("train_distribution")[self.simulation_number]
        l_beam = end-start
        
        n_neurons = self.domain.get("n_neurons")[self.simulation_number]
        n_hiddens = self.domain.get("n_hiddens")[self.simulation_number]
        layer_size = [1] + [n_neurons] * n_hiddens + [1]
        activation = self.domain.get("activation")[self.simulation_number]
        initializer = self.domain.get("initializer")[self.simulation_number]
        
        optimizer_name = self.domain.get("optimizer_name")[self.simulation_number]
        learning_rate = self.domain.get("learning_rate")[self.simulation_number]
        metric = self.result_description.get("output_options").get("metric")
        
        iterations = self.domain.get("iterations")[self.simulation_number]
        
        model_type = self.driver["driver_options"].get("model_type","fixed")         
        
        printed_targets = self.print_targets()
        
        file_content = f'''
import deepxde as dde
import numpy as np

model_type = '{model_type}'
pressure = {pressure}
young_modulus = {young_modulus}
inertia = {inertia}
start = {start}
end = {end}
n_domain = {n_domain}
n_boundary = {n_boundary}
train_distribution = '{train_distribution}'
l_beam = {l_beam}

layer_size = {layer_size}
activation = '{activation}'
initializer = '{initializer}'

optimizer_name = '{optimizer_name}'
learning_rate = {learning_rate}
metric = ['{metric}']

plot_solution = {self.result_description["plot_solution"]}

iterations = {iterations}

geom = dde.geometry.Interval(start, end)

def ddy(x, y):
    return dde.grad.hessian(y, x)

def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], start)

def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], end)

if model_type == "fixed":
    
    def p(x):
        return pressure
    
    def analytical_solution(x):
        return -pressure*x**2/(24*young_modulus*inertia)*((x-l_beam)**2)

    bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
    bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
    bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
    bc4 = dde.NeumannBC(geom, lambda x: 0, boundary_r)

elif model_type == "cantilever":
    
    def p(x):
        return pressure*x/l_beam
    
    def analytical_solution(x):
        return -pressure*x**2/(120*young_modulus*inertia)*(20*l_beam**3 - 10*l_beam**2*x + x**3)

    bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
    bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
    bc3 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
    bc4 = dde.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)

elif model_type == "simply_supported":
    
    def p(x):
        return 4*pressure*(x-l_beam)**2
    
    def analytical_solution(x):
        return -pressure*x/(90*young_modulus*inertia)*(x**5 - 6*x**4*l_beam + 15*x**3*l_beam**2 - 15*x**2*l_beam**3 + 5*l_beam**2)

    bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
    bc2 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_l)
    bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
    bc4 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)

def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return dy_xxxx + p(x)
    
data = dde.data.PDE(
    geom,
    pde,
    [bc1, bc2, bc3, bc4],
    num_domain=n_domain,
    num_boundary=n_boundary,
    solution=analytical_solution,
    num_test=100,
    train_distribution=train_distribution
)

net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(optimizer_name, lr=learning_rate, metrics=metric)

losshistory, train_state = model.train(iterations=iterations)

{printed_targets}

if plot_solution:
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)
'''
        model_file_name = self.output_dir + "/" + self.pinn_model_file_name + ".py"
        
        f = open(model_file_name, "w")
        f.write(file_content)
        f.close()
        
        return model_file_name