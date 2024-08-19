import numpy as np
from time import time
import math
from .optimizers import gradient_free_optimizer, tensorflow_optimizer, newtonian, adam, sgd
from .circuit import Circuit
from .loss import Loss


class VQA(object):

    def __init__(self, name_library, name_backend,  device ='cuda'):
        self.name_library = name_library
        self.name_backend = name_backend
        self.parameter_iteration = []
        self.device = device
        self.hot_start_state = None


    def set_circuit(self, qubits=None, layer=None, entanglement=None, rotation=None, connectivity=None,compression=None, hot_solution =None, maps=None, identity_start=True, seed=0):
        self.qubits = qubits
        self.layer = layer
        self.identity_start = identity_start
        circuit = Circuit(qubits, layer, entanglement, rotation, connectivity, name_library=self.name_library, device=self.device)
        if hot_solution is not None:
            self.hot_start_state = circuit.prepare_state(compression,maps, data=hot_solution)
        if identity_start:
            circuit.identity_start(seed)
        else:
            circuit.compile_circuit()
        self.circuit = circuit.get_circuit()
        if self.name_library == 'qibo':
            self.num_params = len(self.circuit.get_parameters(format='flatlist'))
        elif self.name_library == 'qiskit' or self.name_library == 'qiskit_IONQ':
            self.num_params = self.circuit.num_parameters
        elif self.name_library == 'tensorly-quantum':
            self.num_params = len([circuit.state_dict()[i].item() for i in circuit.state_dict()])



    def set_loss(self, loss_shape='PCE', graph=None,update = False, update_args=None, shots = None ):
        self.loss_shape = loss_shape
        self.loss = Loss(name_backend=self.name_backend, name_library=self.name_library,
                                            circuit=self.circuit, qubits=self.qubits, device=self.device, shots=shots, hot_start_state=self.hot_start_state)
        self.loss.set_problem( graph=graph)
        self.loss.set_dynamic_update(update_args , update)
        self.update_args = update_args
        self.graph = graph

    def set_measurement(self, expectation_map=None, expectations_method=None, encoding_args=None, case=None):
        self.loss.set_measurement(expectation_map=expectation_map, expectations_method=expectations_method,
        encoding_args=encoding_args, case=case)

    def _callback(self, x: int) -> None:
        self.parameter_iteration.append(x)

    def minimize(self, method= 'Powell', bounds= None, constraints=(), tol=None,
                 options=None, initial_state=None, hypermapper=None):
        if initial_state is None:
            if self.identity_start:
                initial_state = np.array(self.circuit.get_parameters(format='flatlist'))
            else:
                if self.hot_start_state is not None:
                    initial_state = np.random.normal(0, 0.1, self.num_params)
                else:
                    initial_state = np.random.uniform(-1, 1, self.num_params)*np.pi

            bounds = [(-np.pi, np.pi) for i in range(len(initial_state))]
        gradient_free_optimizers_list = ['HillClimbing', 'StochasticHillClimbing', 'RepulsingHillClimbing',
                                         'SimulatedAnnealingClimbing',
                                         'DownhillSimplexOptimization', 'RandomSearch', 'GridSearch',
                                         'RandomRestartHillClimbing', 'RandomAnnealing', 'PatternSearch',
                                         'PowellsMethod', 'ParallelTempering', 'ParticleSwarmOptimization',
                                         'SpiralOptimization', 'EvolutionStrategy', 'BayesianOptimization',
                                         'LipschitzOptimization', 'DIRECTalgorithm', 'TreeofParzenEstimators',
                                         'ForestOptimize']

        if self.name_backend == "tensorflow":
            import tensorflow as tf
            my_time = time()
            result, parameters, epochs = tensorflow_optimizer(self.loss.get_loss(self.loss_shape), tf.convert_to_tensor(initial_state),self.qubits,self.update_args['result_exact'], self.update_args['instance'], self.update_args['trial'],  options=options)
            timing = time() - my_time
            final_solution = self.loss.return_final_solution()
            loss_ratio = self.loss.get_ratio()
            best_loss_value = self.loss.best_loss_value
        else:
            if method == 'Adam':
                my_time = time()
                result, parameters, nepochs = adam(self.loss.get_loss(self.loss_shape), derivative=self.loss.get_loss_gradient(self.loss_shape), starting_point=initial_state,n_iter=10000)
                timing = time() - my_time
                epochs = nepochs
                final_solution = np.squeeze(self.loss.return_final_solution())
                loss_ratio = self.loss.get_ratio()
                best_loss_value = self.loss.best_loss_value
            elif method == 'SGD':
                my_time = time()
                result, parameters, nepochs = sgd(self.loss.get_loss(self.loss_shape), self.loss.get_loss_gradient(self.loss_shape), initial_state,
                                                                                 n_iter=10000)
                timing = time() - my_time
                epochs = nepochs
                final_solution = np.squeeze(self.loss.return_final_solution()._value)
                loss_ratio = self.loss.get_ratio()[0]._value
                best_loss_value = result[0]
            elif method =='Hypermapper':
                import pandas as pd
                from hypermapper import optimizer
                import json
                epochs = 650
                scenario = {}
                scenario["application_name"] = "loss"
                scenario["optimization_objectives"] = ["value"]
                scenario["optimization_iterations"] = epochs
                scenario["optimization_method"]= "bayesian_optimization"
                scenario["design_of_experiment"]= { "doe_type": "random sampling","number_of_samples": 100}
                scenario["models"] = {}
                scenario["models"]["model"] = "gaussian_process"
                scenario["input_parameters"] = {}
                for i in range(len(initial_state)):
                    temporary = {}
                    temporary["parameter_type"] = "real"
                    temporary["values"] = [-math.pi, math.pi]
                    # temporary["prior"] = "gaussian"
                    scenario["input_parameters"][f"x{i}"] = temporary

                with open("loss_scenario.json", "w") as scenario_file:
                    json.dump(scenario, scenario_file, indent=4)
                my_time = time()
                optimizer.optimize("loss_scenario.json",self.loss.get_loss(self.loss_shape))
                timing = time() - my_time
                variables = [f'x{i}' for i in range(len(initial_state))]
                variables.append('value')
                sampled_points = pd.read_csv("loss_output_samples.csv", usecols=variables)
                parameters = sampled_points.iloc[sampled_points['value'].idxmin()].values.flatten().tolist()
                parameters.pop()
                parameters = np.array(parameters)
                final_solution = np.array([i[0] for i in self.loss.return_final_solution()])
                print(final_solution)
                loss_ratio = self.loss.get_ratio()[0]
                best_loss_value= self.loss.best_loss_value[0]
            elif method in gradient_free_optimizers_list:
                bounds = [(-np.pi, np.pi) for i in range(len(initial_state))]
                bounds = {f'x{i}': np.arange(bounds[i][0], bounds[i][1], 0.001) for i in range(len(bounds))}
                my_time = time()
                result = gradient_free_optimizer(method, bounds, 20, self.loss.get_loss(self.loss_shape))
                timing = time() - my_time
                result, parameters,epochs  = -result.best_score, np.array(
                    list(result.best_para.values())), len(result.search_data)
                final_solution = self.loss.return_final_solution()
                loss_ratio = self.loss.get_ratio()
                best_loss_value= self.loss.best_loss_value
            else:
                if method == 'COBYLA':
                    bounds = [[-np.pi, np.pi] for i in range(len(initial_state))]
                    cons = []
                    for factor in range(len(bounds)):
                        lower, upper = bounds[factor]
                        l = {'type': 'ineq',
                             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
                        u = {'type': 'ineq',
                             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
                        cons.append(l)
                        cons.append(u)
                    bounds = None
                else:
                    pass
                    # bounds = [(0, 4*np.pi) for i in range(len(initial_state))]
                options = {'maxiter':1000000}
                my_time = time()
                result, parameters, extra = newtonian(self.loss.get_loss(self.loss_shape), initial_state,
                                                                                    method=method, jac=self.loss.get_loss_gradient(self.loss_shape),
                                                                                    bounds=bounds,
                                                                                    constraints=constraints,
                                                                                    tol=0.01, callback=self._callback,
                                                                                    options=options)
                timing = time() - my_time
                epochs =extra['nit']
                final_solution =  np.squeeze(self.loss.return_final_solution())
                loss_ratio = self.loss.get_ratio()
                best_loss_value= self.loss.best_loss_value

        return  initial_state, parameters, epochs, timing, loss_ratio, best_loss_value, final_solution

