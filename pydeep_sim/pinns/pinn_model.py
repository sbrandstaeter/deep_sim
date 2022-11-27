"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import tensorflow as tf
from deepxde.metrics import accuracy, l2_relative_error, nanl2_relative_error, mean_l2_relative_error, _absolute_percentage_error, mean_absolute_percentage_error, max_absolute_percentage_error, absolute_percentage_error_std, mean_squared_error

class PinnModel():

    def __init__(self, output_dir, driver, domain, simulation_number):
        self.output_dir = output_dir
        self.driver = driver
        self.domain = domain
        self.simulation_number = simulation_number
        self.analytical_solution = None

    def build_data(self):
        if self.driver["driver_name"] == "euler_beam":
            input_dim, output_dim, data = self.euler_beam()
        else:
            raise NameError("The chosen driver is not available!")
        return input_dim, output_dim, data
    
    def build_network(self, input_dim, output_dim):
        n_neurons = self.domain.get("n_neurons")[self.simulation_number]
        n_hiddens = self.domain.get("n_hiddens")[self.simulation_number]
        
        layer_size = [input_dim] + [n_neurons] * n_hiddens + [output_dim]
        activation = self.domain.get("activation")[self.simulation_number]
        initializer = self.domain.get("initializer")[self.simulation_number]
        
        net = dde.maps.FNN(layer_size, activation, initializer)
        
        return net
    
    def build_optimizer(self,model):
        
        optimizer_name = self.domain.get("optimizer_name")[self.simulation_number]
        learning_rate = self.domain.get("learning_rate")[self.simulation_number]
        
        model.compile(optimizer_name, lr=learning_rate) # , metrics=["l2 relative error"]
        
        return model
    
    def build_model(self):
        
        input_dim, output_dim, data = self.build_data()
        net = self.build_network(input_dim, output_dim)
        
        model = dde.Model(data, net)
        model = self.build_optimizer(model) 
        
        return model
    
    def run(self, model):
        iterations = self.domain.get("iterations")[self.simulation_number]
        
        losshistory, train_state = model.train(iterations=iterations)
        
        return losshistory, train_state, model
    
    def model_output(self, losshistory, train_state, model):
        
        metric_dict = {
            "accuracy": accuracy,
            "l2 relative error": l2_relative_error,
            "nanl2 relative error": nanl2_relative_error,
            "mean l2 relative error": mean_l2_relative_error,
            "mean squared error": mean_squared_error,
            "MSE": mean_squared_error,
            "mse": mean_squared_error,
            "MAPE": mean_absolute_percentage_error,
            "max APE": max_absolute_percentage_error,
            "APE SD": absolute_percentage_error_std,
        }
        
        model_output = self.driver["driver_options"].get("model_output")
        loss_type = model_output.get("loss_type")
        approach = model_output.get("approach")
        metric_name = model_output.get("metric")
        
        
        if loss_type == "test":
            final_loss = losshistory.loss_test[-1]
        elif loss_type == "train":    
            final_loss = losshistory.loss_test[-1]
        else:
            final_loss = None
        
        if approach == "total":
            final_loss = final_loss.sum()
        
        if metric_name:
            metric = metric_dict[metric_name]
            xtest = model.data.geom.random_points(100)
            utrue = self.analytical_solution(xtest)
            uhat = model.predict(xtest)
            
            model_accuracy = metric(utrue, uhat)
        else:
            model_accuracy = None
            
        return final_loss, model_accuracy
        
    def euler_beam(self):
        
        pressure = self.domain.get("pressure")[self.simulation_number]
        young_modulus = self.domain.get("young_modulus")[self.simulation_number]
        inertia = self.domain.get("inertia")[self.simulation_number]
        start = self.domain.get("start")[self.simulation_number]
        end = self.domain.get("end")[self.simulation_number]
        n_domain = self.domain.get("n_domain")[self.simulation_number]
        n_boundary = self.domain.get("n_boundary")[self.simulation_number]
        
        geom = dde.geometry.Interval(start, end)
        l_beam = end-start
        
        model_type = self.driver["driver_options"].get("model_type","fixed")
        
        def ddy(x, y):
            return dde.grad.hessian(y, x)

        def dddy(x, y):
            return dde.grad.jacobian(ddy(x, y), x)

        def pde(x, y):
            dy_xx = ddy(x, y)
            dy_xxxx = dde.grad.hessian(dy_xx, x)
            return dy_xxxx + p(x)

        def boundary_l(x, on_boundary):
            return on_boundary and np.isclose(x[0], start)

        def boundary_r(x, on_boundary):
            return on_boundary and np.isclose(x[0], end)

        if model_type == "fixed":
            
            def p(x):
                return pressure
            
            def analytical_solution(x):
                return -pressure*x**2/(24*young_modulus*inertia)*((x-l_beam)**2)
            
            self.analytical_solution = analytical_solution

            bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
            bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
            bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
            bc4 = dde.NeumannBC(geom, lambda x: 0, boundary_r)
        
        elif model_type == "cantilever":
            
            def p(x):
                return pressure*x/l_beam
            
            def analytical_solution(x):
                return -pressure*x**2/(120*young_modulus*inertia)*(20*l_beam**3 - 10*l_beam**2*x + x**3)
            
            self.analytical_solution = analytical_solution

            bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
            bc2 = dde.NeumannBC(geom, lambda x: 0, boundary_l)
            bc3 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
            bc4 = dde.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)
        
        elif model_type == "simply_supported":
            
            def p(x):
                return 4*pressure*(x-l_beam)**2
            
            def analytical_solution(x):
                return -pressure*x/(90*young_modulus*inertia)*(x**5 - 6*x**4*l_beam + 15*x**3*l_beam**2 - 15*x**2*l_beam**3 + 5*l_beam**2)
            
            self.analytical_solution = analytical_solution

            bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_l)
            bc2 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_l)
            bc3 = dde.DirichletBC(geom, lambda x: 0, boundary_r)
            bc4 = dde.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
            

        data = dde.data.PDE(
            geom,
            pde,
            [bc1, bc2, bc3, bc4],
            num_domain=n_domain,
            num_boundary=n_boundary,
            solution=analytical_solution,
            num_test=100
        )
        
        input_dim = 1
        output_dim = 1
        
        return input_dim, output_dim, data
    
    