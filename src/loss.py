# ==============================================================================
# Copyright 2023-* Marco Sciorilli. All Rights Reserved.
# Copyright 2023-* QRC @ Technology Innovation Institute of Abu Dhabi. All Rights Reserved.
# ==============================================================================


from math import ceil
from time import time
import networkx as nx
import autograd.numpy as np

from .measurement import Measurement
from .newgraph import RandomGraphs
from .datamanager import insert_value_table
from .utilities import local_search, _round, _obj_value

import tlquantum as tlq
import tensorflow as tf




class Loss(object):

    def __init__(self, name_library, name_backend, circuit, qubits, device='cuda', shots=None, hot_start_state=None):
        self.name_backend = name_backend
        self.name_library = name_library
        self.circuit = circuit
        self.qubits = qubits
        self.loss_ratio = 0
        self.best_loss_value = 1000000000
        self.update = False
        self.epoch = 0
        self.device = device
        self.shots = shots
        self.hot_start_state = hot_start_state

    def set_problem(self, graph=None, spins_number=None):
        self.activation_function = np.tanh
        if graph:
            self.graph = graph
            self.adj_matrix = RandomGraphs._graph_to_dict(self.graph)
            self.nedges = len(self.adj_matrix)
            self.weighted = False
            for _, _, weight in graph.edges(data='weight', default=1):
                if weight != 1:
                    self.weighted = True
                    disposable_graph = graph.copy()
                    for u, v, d in disposable_graph.edges(data=True):
                        if d['weight'] < 0:
                            d['weight'] = 0
                    self.total_weight = disposable_graph.size(weight="weight")
                    self.minimum_spanning_tree = nx.minimum_spanning_tree(disposable_graph, weight='weight').size(
                        weight="weight")
                    break
            self.spins_number = graph.number_of_nodes()
        else:
            self.spins_number= spins_number

    def set_measurement(self, expectation_map=None, expectations_method=None, encoding_args=None, case=None):
        self.expectation_map = expectation_map
        self.expectations_method = expectations_method
        self.encoding_args = encoding_args
        if encoding_args is not None:
            self.encoding_args.append(self.spins_number)
        self.case = case

    def set_dynamic_update(self, args, update):
        self.kind = args['kind']
        self.method = args['method']
        self.instance = args['instance']
        self.trial = args['trial']
        self.layer = args['layer']
        self.spins_number = args['spins_number']
        self.optimization = args['optimization']
        self.activation_function_name = args['activation_function_name']
        self.compression = args['compression']
        self.pauli_string_length = args['pauli_string_length']
        self.graph_kind = args['graph_kind']
        self.initial_parameters = args['initial_parameters']
        self.hyperparameters = args['hyperparameters']
        self.entanglement = args['entanglement']
        self.rotation = args['rotation']
        self.connectivity = args['connectivity']
        self.loss_name = args['loss_name']
        self.result_exact = args['result_exact']
        self.database_name = args['database_name']
        self.database_name = args['database_name']
        self.time = args['time']
        self.update = update

    def get_ratio(self):
        return self.loss_ratio

    def get_loss(self, loss_shape):
        if 'PCE' in loss_shape:
            return self.PCE_loss()

    def get_loss_gradient(self, loss_shape):
        if 'PCE' in loss_shape:
            return self.PCE_loss_gradient()


    def get_expectation(self, params=None):
        import numpy as np
        if self.name_backend == 'tensorflow':
            self.params = params.numpy()
        elif self.name_library == 'tensorly-quantum':
            self.expectation_map = self.circuit
            self.params = np.array([i[0].item() for i in list(self.circuit.parameters())])
        else:
            self.params = params

        expects = Measurement(self.qubits, run_result=self.get_state(params), method=self.expectations_method,
                              name_library=self.name_library, name_backend=self.name_backend, case=self.case, state_preparation= self.hot_start_state)
        spin_mapping_expectation = expects.get_expectations_values(expectation_map=self.expectation_map,
                                                                   arg=self.encoding_args,
                                                                   observale_number=ceil(self.spins_number/3))
        return spin_mapping_expectation

    def get_state(self, params):

        if self.name_library == 'qibo':
            params = list(params)
            self.circuit.set_parameters(params)
            if self.name_backend == 'tensorflow':
                final_state = self.circuit.execute().state()
            else:

                final_state = self.circuit(self.hot_start_state).state()

        elif self.name_library == 'qiskit':
            from qiskit import execute
            circuit = self.circuit.assign_parameters(params)
            job = execute(circuit, self.name_backend)
            result = job.result()
            final_state = result.get_statevector()
        elif self.name_library == 'tensorly-quantum':
            ncontraq = 2
            state = tlq.spins_to_tt_state([0 for i in range(self.qubits)], device=self.device)
            state = tlq.qubits_contract(state, ncontraq)
            final_state = state
        return final_state

    def return_final_solution(self):
        if self.name_backend == 'tensorflow':
            unrounded_solution = self.iteration_expets.numpy()
        else:
            unrounded_solution = self.iteration_expets
        return unrounded_solution

    def update_table(self):
        return lambda p: self._table_update(self.get_expectation(p), self._loss_numpy_PCE(p))

    def PCE_loss_gradient(self, type='exact'):
        from autograd import grad
        grad_loss = grad(self._loss_numpy_PCE_grad)
        if type == 'exact':
            loss_grad = lambda p: grad_loss(p)
        return loss_grad



    def PCE_loss(self):
        if self.name_backend == 'tensorflow':
            tf.debugging.set_log_device_placement(True)
            tensor_ad_mat_egdes = []
            tensor_ad_mat_weight = []
            for i in self.adj_matrix:
                tensor_ad_mat_egdes.append([i[0], i[1]])
                tensor_ad_mat_weight.append(self.adj_matrix[i])
            tensor_ad_mat_edges = tf.convert_to_tensor(tensor_ad_mat_egdes)
            tensor_ad_mat_weights = tf.convert_to_tensor(tensor_ad_mat_weight, dtype=tf.float64)
            loss = lambda p: self._loss_tensorflow_PCE(self.get_expectation(p), self.qubits, tensor_ad_mat_edges,
                                                       tensor_ad_mat_weights)
        else:
            loss = lambda p: self._loss_numpy_PCE(p)
        return loss

    def _loss_numpy_PCE(self, params) -> float:
        # expects = self.get_expectation(list(params.values()))
        expects = self.get_expectation(params)
        loss = 0

        for i in self.adj_matrix:
            loss = loss + self.adj_matrix[i] * self.activation_function(
                expects[i[0]] * self.hyperparameters[0]) \
                   * self.activation_function(expects[i[1]] * self.hyperparameters[0])
        penalization = 0

        for i in range(self.spins_number):
            penalization = penalization + self.activation_function(expects[i] * self.hyperparameters[0]) ** 2

        if self.weighted == False:
            penalization = (self.nedges / 2 + (self.spins_number - 1) / 4) * (penalization / self.spins_number) ** 2 * \
                           self.hyperparameters[1]
        else:
            penalization = ((self.total_weight) / 2 + (self.minimum_spanning_tree) / 4) * (
                    penalization / self.spins_number) ** 2 * self.hyperparameters[1]
        if loss + penalization < self.best_loss_value:
            self.loss_ratio = penalization / loss
            self.best_loss_value = loss + penalization
            self.iteration_expets = expects
        print(loss + penalization)

        self.epoch += 1
        if self.update and self.epoch % (50) == 0:
            self._table_update(expects, loss + penalization)

        return (loss+ penalization)


    def _loss_numpy_PCE_grad(self, params) -> float:
        # expects = self.get_expectation(list(params.values()))
        expects = self.get_expectation(params)
        loss = 0

        for i in self.adj_matrix:
            loss = loss + self.adj_matrix[i] * self.activation_function(
                expects[i[0]] * self.hyperparameters[0]) \
                   * self.activation_function(expects[i[1]] * self.hyperparameters[0])
        penalization = 0

        for i in range(self.spins_number):
            penalization = penalization + self.activation_function(expects[i] * self.hyperparameters[0]) ** 2

        if self.weighted == False:
            penalization = (self.nedges / 2 + (self.spins_number - 1) / 4) * (penalization / self.spins_number) ** 2 * \
                           self.hyperparameters[1]
        else:
            penalization = ((self.total_weight) / 2 + (self.minimum_spanning_tree) / 4) * (
                    penalization / self.spins_number) ** 2 * self.hyperparameters[1]


        return (loss+ penalization)

    def _loss_tensorflow_PCE(self, node_mapping_expectation, tensor_ad_mat_edges, tensor_ad_mat_weights):
        import tensorflow as tf
        self.epoch += 1
        first_term = tf.squeeze(self.activation_function(
            tf.constant(self.hyperparameters[0], dtype=tf.float64) *
            tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 0])))

        second_term = tf.squeeze(self.activation_function(
            tf.constant(self.hyperparameters[0], dtype=tf.float64) *
            tf.gather(node_mapping_expectation, tensor_ad_mat_edges[:, 1])))
        penalization = tf.reduce_sum(
            tf.math.square(self.activation_function(self.hyperparameters[0] * node_mapping_expectation[0:self.spins_number]), 2))
        if self.weighted == False:
            penalization = (self.nedges / 2 + (self.spins_number - 1) / 4) * (penalization / self.spins_number) ** 2 * \
                           self.hyperparameters[1]
        else:
            penalization = ((self.total_weight) / 2 + (self.minimum_spanning_tree) / 4) * (
                        penalization / self.spins_number) ** 2 * self.hyperparameters[1]

        loss = tf.math.multiply(tensor_ad_mat_weights, tf.math.multiply(first_term, second_term))
        loss = tf.math.reduce_sum(loss)

        if tf.math.add(loss, penalization) < self.best_loss_value:
            self.loss_ratio = (penalization._numpy()) / (loss._numpy())
            self.best_loss_value = tf.math.add(loss, penalization)
            self.iteration_expets = node_mapping_expectation

        if self.update and self.epoch % 50 == 0:
            self._table_update(node_mapping_expectation, tf.math.add(loss, penalization)._numpy())

        # print(self.epoch, tf.math.add(loss, penalization))
        return tf.math.add(loss, penalization)


    def _table_update(self, unrounded_solution, loss):
        if self.name_backend == 'tensorflow':
            unrounded_solution = unrounded_solution.numpy()
        solution_raw = [_round(i) for i in unrounded_solution]
        max_energy_raw = _obj_value(solution_raw, problem_name=self.loss_name, adjacency_matrix=self.adj_matrix)
        energy_ratio_raw = max_energy_raw / self.result_exact

        max_energy_local, solution_local = local_search(solution_raw, problem_name=self.loss_name,
                                                        adjacency_matrix=self.adj_matrix)
        energy_ratio_local = max_energy_local / self.result_exact
        row = {'kind': self.kind, 'method': self.method, 'instance': str(self.instance), 'trial': self.trial,
               'layer_number': self.layer,
               'spins_number': self.spins_number, 'optimization': self.optimization,'shots':0,
               'activation_function': 'tanh',
               'compression': self.compression, 'pauli_string_length': self.pauli_string_length,
               'graph_kind': self.graph_kind, 'qubits': self.qubits, 'solution_raw': str(solution_raw),
               'solution_local': str(list(solution_local)),
               'unrounded_solution': str(unrounded_solution.tolist()),
               'max_energy_raw': max_energy_raw, 'energy_ratio_raw': energy_ratio_raw,
               'max_energy_local': max_energy_local, 'energy_ratio_local': energy_ratio_local,
               'initial_parameters': 'None', 'parameters': str(list(self.params)),
               'number_parameters': len(self.params), 'hyperparameter': str(self.hyperparameters),
               'epochs': self.epoch, 'time': time() - self.time,
               'loss_ratio': self.loss_ratio, 'entanglement': self.entanglement,
               'rotation': self.rotation, 'connectivity': self.connectivity, 'loss_name': self.loss_name,
               'loss_value': loss}

        insert_value_table(self.database_name, self.database_name, row)
