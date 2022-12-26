"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import tensorflow as tf
import collections
import pandas as pd
from deepxde.metrics import accuracy, l2_relative_error, nanl2_relative_error, mean_l2_relative_error, _absolute_percentage_error, mean_absolute_percentage_error, max_absolute_percentage_error, absolute_percentage_error_std, mean_squared_error

class PinnMultiModel():

    def __init__(self, output_dir, result_description, driver, domain, simulation_number, pinn_model_file_name):
        self.output_dir = output_dir
        self.driver = driver
        self.result_description = result_description
        self.domain = domain
        self.simulation_number = simulation_number
        self.analytical_solution = None
        self.pinn_model_file_name = pinn_model_file_name

    def generate_input_file(self):
        
        if self.driver["driver_name"] == "euler_beam":
            model_file_name = self.euler_beam()
        else:
            raise NameError("The chosen driver is not available!")

        return model_file_name
    
    def model_output(self, output_file):
        '''
        Model post-processing.
        
        Parameters
        ----------
        losshistory : DeepXDE object
            contains the loss history
        train_state : DeepXDE object
            contains the state of the training process
        model : tf object 
            trained model
        result_description : dict 
            result description read from input (json) file
        simulation_number : int 
            current simulation number
        '''
        
        # initialize the targets and the features 
        targets = collections.OrderedDict()
        features = collections.OrderedDict()
        
        output_options = self.result_description.get("output_options")
        loss_type = output_options.get("loss_type")
        approach = output_options.get("approach")
        metric_name = output_options.get("metric")
        
        file_base = open(output_file,'r')
        
        def initialize_flags():
            train_flag = False
            test_flag = False
            metric_flag = False
            return train_flag, test_flag, metric_flag
        train_flag, test_flag, metric_flag = initialize_flags()
        
        for line in file_base:
            if train_flag:
                train_loss = np.array([float(i) for i in line.strip().replace("[","").replace("]","").split()])
                train_flag, test_flag, metric_flag = initialize_flags()
            elif test_flag:
                test_loss = np.array([float(i) for i in line.strip().replace("[","").replace("]","").split()])
                train_flag, test_flag, metric_flag = initialize_flags()
            elif metric_flag:
                metric = np.array([float(i) for i in line.strip().replace("[","").replace("]","").split()])
                train_flag, test_flag, metric_flag = initialize_flags()
            if "Final train loss" in line:
                train_flag = True
            elif "Final test loss" in line:
                test_flag = True
            elif "Final metric" in line:
                metric_flag = True
            
                
        if loss_type == "test":
            final_loss = test_loss
            if approach == "total":
                final_loss = final_loss.sum()
            targets["loss"] = final_loss
        elif loss_type == "train":    
            final_loss = train_loss
            if approach == "total":
                final_loss = final_loss.sum()
            targets["loss"] = final_loss
        
        if metric_name:
            targets["model_accuracy"] = metric.sum()
        
        for feature in self.result_description["features"]:
            features[feature] = self.domain[feature][self.simulation_number]
        
        features.update(targets)
        df = pd.DataFrame.from_dict([features])
        
        for key, _ in targets.items():
            df[key] = df[key].map(lambda x: '%.4e' % x)
            
        file_name = self.output_dir + "/" + self.result_description.get("output_file_name", "default.dat")
        
        if self.simulation_number == 0:
            df.to_csv(file_name, index=False, sep="\t", float_format='%.5f')
        else:
            df.to_csv(file_name, mode="a", index=False, header=False, sep="\t", float_format='%.5f')
    
    def print_targets(self):
        targets = '''
print("Final train loss")
print(losshistory.loss_train[-1])
print("Final test loss")
print(losshistory.loss_test[-1])
print("Final metric")
print(losshistory.metrics_test[-1])
        '''
        return targets
        
    def euler_beam(self):
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