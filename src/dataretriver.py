from typing import Tuple
import qibo
import math
import multiprocessing as mp
import numpy as np
from time import time
import pickle
import os
from .newgraph import RandomGraphs
from .VQA import VQA
from .measurement import Measurement
from .utilities import solve_quadratic,_get_exact_solution,_round, _obj_value, local_search, _cut_value
from .datamanager import insert_value_table, read_data

class Benchmarker(object):

    def __init__(self, method: str = 'simulation', kind: str = 'VQA', graph_dict: dict = None, spins_number: int = 18, starting= 0,
                 ending= 1, trials: int = 1,
                 graph_kind: str = 'indexed', activation_function: callable = 'None', hyperparameters: list = 'None',
                 database_name: str = 'MaxCutDatabase', shots =None, loss_name = "PCE") -> None:
        self.method = method
        self.kind = kind
        self.spins_number = spins_number
        self.starting = starting
        self.ending = ending
        self.trials = trials
        self.graph_dict = graph_dict
        self.graph_kind = graph_kind
        self.activation_function = activation_function
        self.hyperparameters = hyperparameters
        self.database_name = database_name
        self.shots = shots
        self.loss_name = loss_name



    def set_circuit(self, layer_number, entanglement, rotation, connectivity, optimization="COBYLA", initial_parameters=None, name_device=None, name_backend=None, name_library='qibo'):
        self.name_device = name_device
        self.name_backend = name_backend
        self.name_library = name_library
        self.layer_number = layer_number
        self.optimization = optimization
        self.initial_parameters = initial_parameters
        self.entanglement = entanglement
        self.rotation = rotation
        self.connectivity = connectivity
        if self.name_library == 'qibo':
            import qibo
            if self.name_device is not None:
                qibo.set_backend(f'{self.name_backend}', platform=self.name_device)
            else:
                qibo.set_backend(f'{self.name_backend}')

    def set_encoding(self, ratio_total_words: float = 'None', pauli_string_length: int = 'None', compression: int = None, lower_order_terms= None, shuffle: bool = False, same_letter: bool = True, qubits: int = None):
        self.ratio_total_words = ratio_total_words
        self.pauli_string_length = pauli_string_length
        self.compression = compression
        self.lower_order_terms = lower_order_terms
        self.shuffle = shuffle
        self.same_letter = same_letter
        self.qubits = qubits
        if self.compression is not None:
            if self.qubits is None:
                if self.lower_order_terms:
                    self.qubits = math.ceil(max(solve_quadratic(1, 1, -2 / 3 * self.spins_number)))
                else:
                    self.qubits = math.ceil(max(solve_quadratic(1, -1, -2 / 3 * self.spins_number)))
            else:
                self.qubits = qubits
            self.pauli_string_length = self.qubits
        elif self.qubits:
            self.qubits = qubits

        else:
            self.qubits = int(np.ceil(self.spins_number / round((4 ** self.pauli_string_length - 1) * self.ratio_total_words)) * self.pauli_string_length)

    def set_expectations_method(self, expectation_method='basis_rotation', expectation_map = None, state_preparation =True):
        self.expectation_method = expectation_method
        self.state_preparation = state_preparation
        if self.name_library == 'tensorly-quantum':
            self.expectation_map = None
        elif expectation_map is None and self.method== 'exact':
            if state_preparation:
                self.name_file = f'./maps_{self.qubits}_' + f'{expectation_method}_lower_{self.lower_order_terms}_upper_{self.compression}_' + f'{self.name_backend}_' + f'{self.name_library}_hot_start.pkl'

            else:
                self.name_file = f'./maps_{self.qubits}_'+f'{expectation_method}_lower_{self.lower_order_terms}_upper_{self.compression}_'+f'{self.name_backend}_'+f'{self.name_library}.pkl'
            if os.path.isfile(self.name_file):
                afile = open(self.name_file, 'rb')
                self.expectation_map = pickle.load(afile)
                afile.close()
            else:
                exp = Measurement(self.qubits, method=self.expectation_method, name_library=self.name_library, name_backend=self.name_backend, state_preparation=state_preparation)
                maps_dict = exp.get_observables_map(self.lower_order_terms, self.compression)
                afile = open(self.name_file, 'wb')
                pickle.dump(maps_dict, afile)
                afile.close()
                self.expectation_map = maps_dict
        else:
            self.expectation_map= expectation_map

    def run(self, multiprocessing,update = False):
        self.update =update
        # Run the process in serial or in parallel depending on the user choice
        if multiprocessing is True and self.name_device =='cpu':
            self._eigensolver_evaluater_parallel()
        elif multiprocessing is True and self.name_device =='gpu':
            self._eigensolver_evaluater_parallel_gpu()
        else:
            self._eigensolver_evaluater_serial()

    def _eigensolver_evaluater_parallel(self) -> None:
        process_number = len(os.sched_getaffinity(0))
        pool = mp.Pool(process_number)
        if self.shots is None:
            shots = [None]
        else:
            shots = self.shots

        if self.graph_dict is not None:
            [pool.apply_async(self._single_instance_evaluation, (0, trial,layer,(graph, self.graph_dict[graph]),shot))  for
            layer in self.layer_number for trial in range(self.trials) for graph in self.graph_dict for shot in shots]

        else:
            [pool.apply_async(self._single_instance_evaluation, (instance, trial, layer, shot)) for layer in self.layer_number for trial in range(self.trials) for instance in range(self.starting, self.ending) for shot in shots]
        pool.close()
        pool.join()



    def _gpu_process(self, queue, instance, trial, graph_dict, layer) -> None:
        gpu_id = queue.get()
        if self.name_backend =='tensorflow':
            device = f'/gpu:{gpu_id}'
        else:
            device = f'cuda:{gpu_id}'
        try:
            self._single_instance_evaluation(instance, trial, graph_dict, layer, device)
        finally:
            queue.put(gpu_id)

    def _eigensolver_evaluater_parallel_gpu(self) -> None:
        """
        Function which carry on the benchmark in a multiprocess framework, running an instance per process
        """
        import torch.multiprocessing as tcmp
        import torch as tc
        NUM_GPUS = tc.cuda.device_count()
        PROC_PER_GPU = 1
        m = tcmp.Manager()
        queue = m.Queue()
        for gpu_ids in range(NUM_GPUS):
            for _ in range(PROC_PER_GPU):
                queue.put(gpu_ids)
        pool = tcmp.Pool(processes=NUM_GPUS * PROC_PER_GPU)
        if self.graph_dict is not None:
            [pool.apply_async(self._gpu_process, (queue, 0, trial, (graph, self.graph_dict[graph]), layer)) for
             layer in
             self.layer_number for graph in self.graph_dict for trial in range(self.starting, self.ending)]
        else:
            [pool.apply_async(self._gpu_process, (queue, instance, trial, self.graph_dict, layer)) for layer in
             self.layer_number for instance in
             range(self.starting, self.ending) for trial in range(self.starting, self.ending)]

        pool.close()
        pool.join()

    def _eigensolver_evaluater_serial(self) -> None:
        """
        Function which carry on the benchmark in a serial framework, running one instance at a time
        """
        if self.shots is None:
            shots = [None]
        else:
            shots = self.shots

        if self.graph_dict is not None:
            for layer in self.layer_number:
                for graph in self.graph_dict:
                    for trial in range(self.trials):
                        for shot in shots:
                            self._single_instance_evaluation(0, trial, layer, graph=(graph, self.graph_dict[graph]), shot=shot)
        else:
            for layer in self.layer_number:
                for trial in range( self.trials):
                    for instance in range(self.starting, self.ending):
                        for shot in shots:
                            self._single_instance_evaluation(instance, trial, layer, None, shot)


    def _single_instance_evaluation(self, instance, trial, layer, graph=None, shot=None,  device='cuda'):
        if self.name_backend == 'torch':
            import torch as tc
            tc.device(device)
        elif self.name_library == 'qibo':
            qibo.set_backend(self.name_backend)

        if graph is None:
            graph, instance = self.do_graph(instance)
            instance_name = str(instance)
            adj_matrix = RandomGraphs._graph_to_dict(graph)
        else:
            instance_name = graph[0]
            graph = graph[1]
            adj_matrix = RandomGraphs._graph_to_dict(graph)
        result_exact, hot_start = _get_exact_solution(self.loss_name,graph)
        if result_exact == 0.0:
            return

        if self.kind == 'VQA':
            np.random.seed(trial + instance)
            if self.method == 'exact' or 'shots-simulation':
                if self.state_preparation:
                    import json

                    # temporary_approx = read_data('baseline_100', 'baseline_100', ['solution_local'],
                    #                              {'instance': instance_name, 'trial': trial})
                    # solution_local = json.loads(temporary_approx[0][0])
                    # result_exact = _cut_value(solution_local, adj_matrix)
                    # # print(instance, trial, _cut_value(solution_local, adj_matrix))
                    # solution_local = solution_local * np.random.uniform(0, 1, 100)
                    # solution_local = np.random.uniform(0, 1, self.spins_number)
                    # print(solution_local)
                    solution_local =hot_start
                    solution_local = solution_local * np.random.uniform(0, 1, self.spins_number)

                else:
                    solution_local = None
                update_args = {'kind':self.kind, 'method':self.method,'instance':instance_name, 'trial':trial,'layer':layer,'spins_number':self.spins_number,'optimization':self.optimization,'activation_function_name':self.activation_function,'compression':self.compression,'pauli_string_length':self.pauli_string_length,'graph_kind':self.graph_kind,'initial_parameters':self.initial_parameters,'hyperparameters':self.hyperparameters, 'entanglement':self.entanglement, 'rotation':self.rotation,'connectivity':self.connectivity,'loss_name':self.loss_name, 'result_exact': result_exact, 'database_name': self.database_name, 'time':time()}
                result = VQA( name_library=self.name_library,  name_backend=self.name_backend, device=device)
                result.set_circuit(self.qubits, layer, self.entanglement, self.rotation, self.connectivity, self.compression, solution_local,maps=self.expectation_map, identity_start=True,seed= trial + instance)
                result.set_loss(loss_shape=self.loss_name, graph=graph, update_args=update_args, update=self.update, shots = shot)


                result.set_measurement(expectation_map=self.expectation_map, expectations_method=self.expectation_method, encoding_args=[self.pauli_string_length, self.compression, self.lower_order_terms, self.shuffle, self.same_letter, self.ratio_total_words, trial+instance], case=self.method)
                result = result.minimize(method=self.optimization, initial_state=self.initial_parameters)

            activation_function_name = 'tanh'

            if self.name_backend == 'torch':
                initial_parameters = result[0]
                parameters = result[1]
            else:
                initial_parameters = result[0].tolist()
                parameters = result[1].tolist()
            number_parameters = len(result[0])
            epochs = result[2]
            timing = result[3]
            loss_ratio = result[4]
            self.loss_value = result[5]
            unrounded_solution = result[6].tolist()
            solution_raw = [_round(i) for i in unrounded_solution]
            max_energy_raw = _obj_value(solution_raw, problem_name=self.loss_name, adjacency_matrix=adj_matrix)
            energy_ratio_raw = max_energy_raw / result_exact

            max_energy_local, solution_local = local_search(solution_raw, problem_name=self.loss_name,
                                                            adjacency_matrix=adj_matrix)
            solution_local = list(solution_local)

            energy_ratio_local = max_energy_local / result_exact


        # Save the instances in the dataset
        instance = instance_name

        row = {'kind': self.kind,'method':self.method, 'instance': str(instance), 'trial': trial, 'layer_number': layer,
               'spins_number': self.spins_number, 'optimization': self.optimization,
               'activation_function': str(activation_function_name),
               'compression': self.compression, 'pauli_string_length': self.pauli_string_length,
               'graph_kind': self.graph_kind, 'qubits': self.qubits,'shots':0,'solution_raw': str(solution_raw),'solution_local': str(solution_local),
               'unrounded_solution': str(unrounded_solution),
               'max_energy_raw': max_energy_raw,  'energy_ratio_raw': energy_ratio_raw,'max_energy_local': max_energy_local, 'energy_ratio_local': energy_ratio_local,
               'initial_parameters': str(initial_parameters), 'parameters': str(parameters),
               'number_parameters': number_parameters, 'hyperparameter': str(self.hyperparameters),
               'epochs': epochs, 'time': timing, 'loss_ratio': loss_ratio,'entanglement' : self.entanglement, 'rotation' : self.rotation,'connectivity' : self.connectivity, 'loss_name': self.loss_name, 'loss_value': self.loss_value}
        insert_value_table(self.database_name, self.database_name, row)
        return row

    def do_graph(self, instance: int) -> Tuple[object, int]:
        """
        Function which, given the user specification, return the wanted genereted graph
        :param instance: Index of the graph to generate.
        :return: The graph.
        """
        true_random_graphs = False
        fully_connected = False
        if self.graph_kind == 'random':
            true_random_graphs = True
        if self.graph_kind == 'fully':
            fully_connected = True
        graph = RandomGraphs(instance, self.spins_number, true_random_graphs, fully_connected=fully_connected).graph
        if true_random_graphs:
            instance = graph.return_index()
        return graph, instance




