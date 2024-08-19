import autograd.numpy as np
import itertools
import tensorflow as tf
import torch as tc
import tlquantum as tlq
from . import circuit

from opt_einsum import contract_expression


class Measurement(object):
    def __init__(self, nqubits, run_result=None, method='basis_rotation', case='simulation', name_library='qibo', name_backend='numpy', shots = 1024, state_preparation=False):
        self.nqubits = nqubits
        self.identity = np.array([[1, 0], [0, 1]])
        self.np_x = np.array([[0, 1], [1, 0]])
        self.np_y = np.array([[0, -1j], [1j, 0]])
        self.np_z = np.array([[1, 0], [0, -1]])
        self.run_result = run_result
        self.method = method
        self.name_library = name_library
        self.name_backend = name_backend
        self.case = case
        self.shots = shots
        self.state_preparation = state_preparation

        if run_result is not None and (self.case== 'exact' or self.case=='shots-simulation' or self.case == 'simulation'):
            self._state_updater(run_result)


    def _state_updater(self,state):
        self.state = state
        if self.name_backend == 'tensorflow':
            self.state_squared = tf.math.abs(tf.math.square(state))
        elif self.name_library == 'tensorly-quantum':
            pass
        else:
            self.state_squared = np.abs(state) ** 2
            self.state_bra = np.transpose(np.conjugate(state))
    def set_run_result(self,run_result):
        self.run_result = run_result


    def get_observables_map(self, lower_order_terms, pauli_word_order):
        if 'basis_rotation' in self.method:
            if self.name_backend == 'tensorflow':
                expectation_map = self.find_mapping_rotation_tensorflow(lower_order_terms, pauli_word_order)
                return expectation_map
            else:
                expectation_map = self.find_mapping_rotation(lower_order_terms, pauli_word_order)
                return expectation_map
        else:
            pass

    def get_expectations_values(self, expectation_map=None,  arg=None, observale_number=None):
        if self.case == 'exact':
            if 'basis_rotation' in self.method:
                expectations_values = self.expectations_rotation(expectation_map,observale_number)
            elif self.method == 'symbolic':
                if self.name_library == 'qibo':
                    expectations_values = self.general_expectations_symbolic(arg)
                elif self.name_library == 'qiskit':
                    raise ('NOT IMPLEMENTED YET')
        elif self.case == 'experiment':
            if self.method == 'basis_rotation':
                expectations_values = self.expectations_from_counts(self.run_result, self.shots, arg[2], arg[2])
        else:
            raise('No other methods')
        return expectations_values

################################################################################################################
## TENSOR CONTRACTION
################################################################################################################
    def tensor_contraction(self, circuit, contractions, two_qubit_ptraces ):
        expectation_values = self.measure_two_qubit_expectation_values(circuit, self.state, contractions, two_qubit_ptraces)
        return expectation_values
################################################################################################################
## BASIS ROTATION METHOD
################################################################################################################
        ################################################################################################################
        ## MAPPING
        ################################################################################################################


    def find_mapping_rotation(self, lower_order_terms, pauli_word_order):
        single_pauli_obs_vec_z = []
        many_pauli_obs_vec = []
        for i in range(self.nqubits):
            vec = np.zeros((1, 2 ** self.nqubits))
            for j in range(0, 2 ** self.nqubits, 2 ** (self.nqubits - 1 - i)):
                vec[0, j:j + 2 ** (self.nqubits - 1 - i)] = (-1) ** (j // 2 ** (self.nqubits - 1 - i))
            single_pauli_obs_vec_z.append(vec.real)

        reset = True
        for i in range(lower_order_terms, pauli_word_order + 1):
            for gate_indices in list(itertools.combinations(list(range(self.nqubits)), i)):
                for ind in range(self.nqubits):
                    if ind in gate_indices:
                        if reset:
                            op = single_pauli_obs_vec_z[ind]
                            reset = False
                        else:
                            op = op * single_pauli_obs_vec_z[ind]
                many_pauli_obs_vec.append(op)
                reset = True
        return many_pauli_obs_vec


        ################################################################################################################
        ## EXPECTATION
        ################################################################################################################

    def expectations_rotation(self, expectation_map, observale_number):
        if expectation_map is None:
            # Defaul is quadratic compression
            expectation_map = self.find_mapping_rotation(2, 2)

        if self.state_preparation is None or False:
            constraints_z = self.expectations_vector_z(expectation_map[:observale_number])
            constraints_x = self.expectations_vector_x_rotation(expectation_map[:observale_number])
            constraints_y = self.expectations_vector_y_rotation(expectation_map[:observale_number])
        else:
            constraints_z = self.expectations_vector_z(expectation_map['z'][:observale_number])
            constraints_x = self.expectations_vector_x_rotation(expectation_map['x'][:observale_number])
            constraints_y = self.expectations_vector_y_rotation(expectation_map['y'][:observale_number])



        if self.name_backend == 'tensorflow':
            constraints = tf.concat([constraints_z, constraints_x, constraints_y], axis=0)
        else:
            constraints = np.concatenate((constraints_z, constraints_x, constraints_y))
        return constraints

    def expectations_vector_x_rotation(self, expectation_map):
        self._state_updater(circuit.Circuit.rotate_x_measure(self.state, self.name_library, self.nqubits))
        return self.expectations_vector_z(expectation_map)

    def expectations_vector_y_rotation(self, expectation_map):
        self._state_updater(circuit.Circuit.rotate_y_measure(self.state,  self.name_library, self.nqubits))
        return self.expectations_vector_z(expectation_map)

    def expectations_vector_z(self, expectation_map):
        if self.name_backend == 'tensorflow':
            constraints = tf.linalg.matvec(expectation_map, self.state_squared)
        else:
            if self.state_preparation is None or False:
                constraints = expectation_map @ self.state_squared
            else:
                constraints = np.sum(self.state_squared[expectation_map[:, 1]] * expectation_map[:, 0], axis=1)

        return constraints


########################################################################################################################
## TENSORFLOW IMPLEMENTATION
########################################################################################################################

    def find_mapping_rotation_tensorflow(self,lower_order_terms, pauli_word_order ):
        single_pauli_obs_vec_z = []
        two_pauli_obs_vec = []
        for i in range(self.nqubits):
            vec = np.zeros((1, 2 ** self.nqubits))
            for j in range(0, 2 ** self.nqubits, 2 ** (self.nqubits - 1 - i)):
                vec[0, j:j + 2 ** (self.nqubits - 1 - i)] = (-1) ** (j // 2 ** (self.nqubits - 1 - i))
            single_pauli_obs_vec_z.append(vec.real)

        reset = True
        for i in range(lower_order_terms, pauli_word_order + 1):
            for gate_indices in list(itertools.combinations(list(range(self.nqubits)), i)):
                for ind in range(self.nqubits):
                    if ind in gate_indices:
                        if reset:
                            op = single_pauli_obs_vec_z[ind]
                            reset = False
                        else:
                            op = op * single_pauli_obs_vec_z[ind]
                two_pauli_obs_vec.append(tf.convert_to_tensor(op[0]))
                reset = True
        return two_pauli_obs_vec


########################################################################################################################
## SYMBOLIC QIBO IMPLEMENTATION
########################################################################################################################
    def general_expectations_symbolic(self, args):
        self.nodes_number = args[7]
        self.pauli_word_length = args[0]
        self.compression = args[1]
        self.lower_order_terms = args[2]
        self.shuffle = args[3]
        self.same_letter = args[4]
        self.ratio_total_words = args[5]
        self.seed = args[6]
        node_mapping = self.encode_nodes(self.nodes_number, self.pauli_word_length,
                                         compression=self.compression, lower_order_terms=self.lower_order_terms,
                                         shuffle=self.shuffle, seed=self.seed, same_letter=self.same_letter)
        node_mapping_expectation = [i.expectation(self.state) for i in node_mapping]
        return node_mapping_expectation

    def encode_nodes(self, num_nodes: int, pauli_string_length: int, ratio_total_words: float = None,
                     compression: int = None,
                     lower_order_terms: bool = False, shuffle: bool = True, seed: int = 0,
                     same_letter: bool = True, print_strings=False):
        """
        Function which save the encodings of the graph nodes in the chosen observables (expressed in symbolic Hamiltonians)
        :param num_nodes: number of nodes in the graph.
        :param pauli_string_length: Maximum length of any pauli word (also considering Identities).
        :param ratio_total_words: ratio of the pauli word to use in the compression among all the feasable ones up to
                                pauli_string_length
        :param compression: Order of the compression to be used (if explicitly expressed)
        :param lower_order_terms: Whether or no also include shorter pauli word.
        :param shuffle: Whether or no shuffle the list of pauli word.
        :param seed: Seed for the shuffling.
        :param same_letter: Whether of no use observables of the same pauli letter
        :return:
        """

        def get_pauli_word(indices, k: object, qubits: int = None):
            """
            Function which, given a pauli word, return the corresponding symbolic hamiltonian
            :param indices: pauli word as a list of integers
            :param k: number of the qubit of the observables
            :param qubits:total number of qubits used
            :return: Symbolic hamiltonian of the pauli word
            """
            from qibo import hamiltonians
            from qibo.symbols import I, X, Y, Z
            # Given the pauli word, write it as a symbolic hamiltonian
            # Generate pauli string corresponding to indices
            # where (0, 1, 2, 3) -> 1XYZ and so on
            pauli_matrices = np.array([I, X, Y, Z])
            word = np.int(1)
            for qubit, i in enumerate(indices):
                word *= pauli_matrices[i](qubit + int(k))

            # If the number of qubits is given, add identity of all the unused qubits
            if qubits:
                qubits_list = list(range(qubits))
                qubits_list.remove(int(k))
                for j in qubits_list:
                    word *= pauli_matrices[0](j)
            return hamiltonians.SymbolicHamiltonian(word)

        # If the compression is not explicity expressed, infer the encoding from pauli_string_length and ratio_total_words
        if compression is None:
            pauli_strings = self._pauli_string_same_letter(pauli_string_length, 2, True, shuffle,
                                                           seed,
                                                           pauli_letters=int(ratio_total_words * 3 + 1))
            num_strings = len(pauli_strings)

            # position i stores string corresponding to the i-th node.
            node_mapping = [
                get_pauli_word(pauli_strings[int(i % num_strings)], pauli_string_length * np.floor(i / num_strings),
                               self.get_num_qubits(num_nodes, pauli_string_length, ratio_total_words)) for i in
                range(num_nodes)]
        else:
            # If the compression is explicitly expressed, get the list of the pauli words (subject to user requirements)
            if same_letter:
                pauli_strings = self._pauli_string_same_letter(pauli_string_length, 2,
                                                               lower_order_terms,
                                                               shuffle, seed)
            else:
                pauli_strings = self._random_pauli_string(pauli_string_length, compression, lower_order_terms,
                                                          shuffle, seed)
            num_strings = len(pauli_strings)
            if print_strings:
                print(pauli_strings)

            # position i stores string corresponding to the i-th node.
            node_mapping = [
                get_pauli_word(pauli_strings[int(i % num_strings)],
                               pauli_string_length * np.floor(i / num_strings))
                for i in range(num_nodes)]
        return node_mapping  # , np.ceil(num_nodes / num_strings)]


########################################################################################################################
## UTILITIES
########################################################################################################################

    def measure_two_qubit_expectation_values(self, circuit, state, contractions, two_qubit_ptraces):
        expectation_values, ev_count = tc.zeros(3 * int(circuit.nqubits * (circuit.nqubits - 1) / 2), ), 0
        for con, tqp in zip(contractions, two_qubit_ptraces):
            rdm = con(*circuit._build_circuit(state))
            rdm_nqubits = int(np.log2(np.prod(rdm.shape)) / 2)
            dims = [2 for i in range(rdm_nqubits)]
            dims = [dims, dims]
            rdm = tlq.DensityTensor(rdm.reshape(sum(dims, [])), dims)
            for indices in tqp:
                two_qubits = rdm.partial_trace(indices)[0]
                expects = self.lightweight_xx_yy_zz(two_qubits)
                expectation_values[ev_count] = expects[0]
                expectation_values[ev_count + int(len(expectation_values)/3)] = expects[1]
                expectation_values[ev_count + int(len(expectation_values)/3*2)] = expects[2]
                ev_count += 1
        return expectation_values

    def lightweight_xx_yy_zz(self, two_qubits_rdm):
        two_qubits_rdm = two_qubits_rdm.real
        zb, zc, xya, xyb = two_qubits_rdm[0, 1, 0, 1], two_qubits_rdm[1, 0, 1, 0], two_qubits_rdm[0, 0, 1, 1], \
                           two_qubits_rdm[1, 0, 0, 1]
        return tc.stack([ 1 - 2 * zb - 2 * zc, 2 * (xya + xyb), 2 * (-xya + xyb)])

    def generate_contraction_path(self, circuit, ptrace_groups, state):
        equations = self.generate_ptrace_equations(circuit.nqsystems, circuit.nlsystems, ptrace_groups)
        shapes = [core.shape for core in circuit._build_circuit(state)]
        return [contract_expression(equation, *shapes) for equation in equations]

    def generate_ptrace_equations(self, nqsystems, nlsystems, ptrace_groups):
        return [tlq.contraction_eq(nqsystems, 2 * nlsystems, kept_inds=indices) for indices in ptrace_groups]

    def return_qubit_pairs(self, ptrace_groups, two_qubit_ptraces, n, ncontraq):
        qubit_pairs = []
        max_group = max(ptrace_groups[0])
        lcsize = ncontraq if n % ncontraq == 0 else n % ncontraq
        for pg, tqp in zip(ptrace_groups, two_qubit_ptraces):
            for index1, index2 in tqp:
                if pg[index2 // ncontraq] == max_group:
                    qubit_pairs.append(
                        (ncontraq * pg[index1 // ncontraq] + index1 % ncontraq, n - lcsize + index2 % ncontraq))
                else:
                    qubit_pairs.append((ncontraq * pg[index1 // ncontraq] + index1 % ncontraq,
                                        ncontraq * pg[index2 // ncontraq] + index2 % ncontraq))
        return qubit_pairs

    def generate_twobody_indices(self, n, ncontraq, m):
        num_groups, ncores, lcsize = int(np.ceil(n / (ncontraq * m))), int(
            np.ceil(n / ncontraq)), ncontraq if n % ncontraq == 0 else n % ncontraq
        index_groups = [list(np.arange(m * k, m * (k + 1))) for k in range(num_groups - 1)] + [
            list(np.arange(m * (num_groups - 1), ncores))]
        group_size, last_group_size = ncontraq * len(index_groups[0]), ncontraq * (len(index_groups[-1]) - 1) + lcsize
        ptrace_groups = [index_groups[index2] + index_groups[index1] for index1 in range(num_groups - 1, 0, -1) for
                         index2
                         in range(index1)]
        two_qubit_ptraces = [[(j, k) for j in range(group_size) for k in range(j + 1, group_size + last_group_size)] for
                             _
                             in range(num_groups - 1)]
        two_qubit_ptraces += [[(j, k) for j in range(group_size) for k in range(group_size, 2 * group_size)] for _ in
                              range(num_groups - 1, len(ptrace_groups))]
        two_qubit_ptraces[0] += [(i, j) for i in range(group_size, group_size + last_group_size) for j in
                                 range(i + 1, group_size + last_group_size)]
        return ptrace_groups, two_qubit_ptraces

    @staticmethod
    def _pauli_string_same_letter(pauli_string_length: int, order: int, lower_order_terms: bool, shuffle: bool = False,
                                  seed: int = 1,
                                  pauli_letters: int = 4) -> list:
        """
        Function which returns a list of all the pauli words satisfying the requirements asked by the user.
        :param pauli_string_length: Maximum length of any pauli word (also considering Identities).
        :param order: Maximum number of qubits actually interested in the pauli word (closely dependent on the compression).
        :param lower_order_terms: Whether or no also include shorter pauli word.
        :param shuffle: Whether or no shuffle the list of pauli word.
        :param seed: Seed for the shuffling.
        :param pauli_letters:Number of pauli letters to be used in the pauli words (obv max 4).
        :return: list of pauli word encoded as tuples of integers.
        """
        # Initialise the list
        pauli_string = []

        # If required, use all pauli words of length up to order
        if lower_order_terms:
            smallest_length = 1
        else:
            smallest_length = order

        # Append all the combinations of given length of same pauli letter
        for i in range(1, pauli_letters):
            for k in range(smallest_length, order + 1):
                comb = itertools.combinations(list(range(pauli_string_length)), k)
                for positions in comb:
                    instance = [0] * pauli_string_length
                    for index in positions:
                        instance[index] = i
                    pauli_string.append(tuple(instance))

        # If required, shuffle the list
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(pauli_string)
        return pauli_string


    @staticmethod
    def _random_pauli_string(pauli_string_length: int, order: int, lower_order_terms: bool, shuffle: bool,
                             seed: int) -> list:
        """
        Same as _pauli_string_same_letter, but picking instead random pauli words among, not only the one of the
        same letter.
        :param pauli_string_length: Maximum length of any pauli word (also considering Identities).
        :param order: Maximum number of qubits actually interested in the pauli word (closely dependent on
                    the compression).
        :param lower_order_terms: Whether or no also include shorter pauli word.
        :param shuffle: Whether or no shuffle the list of pauli word.
        :param seed: Seed for the shuffling.
        :return:
        """
        # Initialise all possibile tuples (pauli letter, qubit)
        pauli_tuples = [(i, j) for i in range(1, 4) for j in range(pauli_string_length)]

        # If required, use all pauli words of length up to order
        if lower_order_terms:
            smallest_length = 1
        else:
            smallest_length = order

        # Create list of all possible combinations of pauli word of given length
        total_combinations = []
        for i in range(smallest_length, order + 1):
            total_combinations = total_combinations + (list(itertools.combinations(pauli_tuples, i)))

        # Create the final list of pauli words
        pauli_string = []
        for comb in total_combinations:
            instance = [0] * pauli_string_length
            for j in comb:
                instance[j[1]] = j[0]
                pauli_string.append(instance)

        # If required, shuffle the list
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(pauli_string)
        return pauli_string

    @staticmethod
    def get_num_qubits(num_nodes: int, pauli_string_length: int, ratio_total_words: float) -> int:
        """
        Function which, given the number of nodes of the graph problem, and the configurations of the compression,
        returns the number of qubits necessary to carry on the algorithm
        :param num_nodes: number of nodes in the graph.
        :param pauli_string_length: maximum length of the pauli words used in the compression.
        :param ratio_total_words: ratio of the pauli word to use in the compression among all the feasable ones up to
                                pauli_string_length
        :return: number of qubits needed in the circuit
        """
        # return the number of qubits necessary
        return int(np.ceil(num_nodes / round((4 ** pauli_string_length - 1) * ratio_total_words)) * pauli_string_length)

