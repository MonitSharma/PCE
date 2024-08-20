# ==============================================================================
# Copyright 2023-* Marco Sciorilli. All Rights Reserved.
# Copyright 2023-* QRC @ Technology Innovation Institute of Abu Dhabi. All Rights Reserved.
# ==============================================================================

import math
import networkx as nx
import numpy as np
from qibo import models, gates
from qiskit.circuit.library import RXGate, RYGate, PhaseGate, RXXGate, CZGate, U3Gate, SdgGate, HGate
from qiskit.circuit import QuantumCircuit, ParameterVector
from math import log2, ceil


class Circuit(object):
    def __init__(self, size: int = 6, p: int = 0, entanglement="CNOT", rotation='U3',
                 connectivity="alternating_closed", name_library='qibo', device='cuda', initial_param=None):
        self.size = size
        self.p = p
        self.entanglement = entanglement
        self.rotation = rotation
        self.connectivity = connectivity
        self.entang_list = []
        self.name_library = name_library
        self.device = device
        self.initial_param = initial_param
        self.circuit_representation = None
    def compile_circuit(self):
        if self.name_library == "qibo":
            if self.circuit_representation is None:
                self.circuit_representation = models.Circuit(self.size)
            circuit = self._qibo_circuit_(self.circuit_representation)
        elif self.name_library == "qiskit":
            c = QuantumCircuit(self.size)
            circuit = self._qiskit_circuit_(c)
        else:
            raise ("Library not implemented yet")
        self.circuit_representation = circuit

    def identity_start(self, seed):
        np.random.seed(seed)
        self.p = ceil(self.p / 2)
        self.compile_circuit()
        self.circuit_representation.set_parameters(
            np.random.uniform(-1, 1, len(self.circuit_representation.get_parameters())) * np.pi)
        inverse_circuit = self.circuit_representation.invert()
        self.circuit_representation = self.circuit_representation + inverse_circuit

    def get_circuit(self):
        return self.circuit_representation

    def _define_connectivity_(self, layer):
        entang_list = []
        layer = math.ceil(layer / 2)
        if self.connectivity == "ladder_open":
            for i in range(self.size - 1):
                entang_list.append((i, i + 1))
        if self.connectivity == "ladder_closed":
            for i in range(self.size - 1):
                entang_list.append((i, i + 1))
            entang_list.append(self.size - 1, 0)
        if self.connectivity == "brickwork_double":
            if self.size % 2 == 0:
                for i in range(0, self.size, 2):
                    entang_list.append((i, i + 1))
                for i in range(1, self.size - 1, 2):
                    entang_list.append((i, i + 1))
            else:
                for i in range(0, self.size - 1, 2):
                    entang_list.append((i, i + 1))
                for i in range(1, self.size, 2):
                    entang_list.append((i, i + 1))
        if self.connectivity == "brickwork_single_open":
            if self.size % 2 == 0:
                if layer % 2 == 0:
                    for i in range(0, self.size, 2):
                        entang_list.append((i, i + 1))
                else:
                    for i in range(1, self.size - 1, 2):
                        entang_list.append((i, i + 1))
            else:
                if layer % 2 == 0:
                    for i in range(0, self.size - 1, 2):
                        entang_list.append((i, i + 1))
                else:
                    for i in range(1, self.size, 2):
                        entang_list.append((i, i + 1))
        if self.connectivity == "brickwork_single_closed":
            if self.size % 2 == 0:
                if layer % 2 == 0:
                    entang_list.append((0, self.size - 1))

                    for i in range(1, self.size - 1, 2):
                        entang_list.append((i, i + 1))
                else:
                    for i in range(0, self.size, 2):
                        entang_list.append((i, i + 1))
            else:
                if layer % 2 == 0:
                    for i in range(0, self.size - 1, 2):
                        entang_list.append((i, i + 1))

                else:
                    for i in range(1, self.size, 2):
                        entang_list.append((i, i + 1))

        if self.connectivity == "brickwork_single_rotating":

            entang_list_unpaired = [q for q in range(layer - 1, self.size + layer - 1)]

            def refit(entang_list_unpaired, size):
                for i in range(len(entang_list_unpaired)):
                    if entang_list_unpaired[i] > size - 1:
                        entang_list_unpaired[i] = entang_list_unpaired[i] - size
                if all(q < size for q in entang_list_unpaired):
                    return entang_list_unpaired
                else:
                    return refit(entang_list_unpaired, size)

            entang_list_unpaired = refit(entang_list_unpaired, self.size)
            for q in range(1, len(entang_list_unpaired), 2):
                entang_list.append((entang_list_unpaired[q - 1], entang_list_unpaired[q]))
        if self.connectivity == 'round_robin':
            def qubits_round_robin(players):
                s = []
                if len(players) % 2 == 1: players = players + [None]
                n = len(players)
                map = list(range(n))
                mid = n // 2
                for i in range(n - 1):
                    l1 = map[:mid]
                    l2 = map[mid:]
                    l2.reverse()
                    round = []
                    for j in range(mid):
                        t1 = players[l1[j]]
                        t2 = players[l2[j]]
                        if j == 0 and i % 2 == 1:
                            round.append((t2, t1))
                        else:
                            round.append((t1, t2))
                    s.append(round)
                    map = map[mid:-1] + map[:mid] + map[-1:]
                return s

            if self.size % 2 == 0:
                lenght = self.size - 1
            else:
                lenght = self.size
            entang_list = qubits_round_robin(list(range(self.size)))[layer % lenght]
        if self.connectivity == "2D_lattice":
            size = math.ceil(math.sqrt(self.size))
            edge_list = []
            if size % 2:
                if layer > 4 and layer % 4:
                    switch = math.ceil(layer / 4)
                    layer = layer % 4
                elif layer > 4:
                    switch = math.ceil(layer / 4)
                    layer = 4
                else:
                    switch = 0
                if layer % 2 == 0 and layer % 4 != 0:
                    for i in range(size):
                        if i % 2 or i + 2 > size:
                            continue
                        else:
                            for j in range(size):
                                edge_list.append(((i, j), (i + 1, j)))
                    for l in range(size):
                        if switch % 2 and switch != 0 and l + 1 < size:
                            if l % 2:
                                edge_list.append(((size - 1, l), (size - 1, l + 1)))
                        else:
                            if l % 2 == 0 and l + 1 < size:
                                edge_list.append(((size - 1, l), (size - 1, l + 1)))
                elif layer % 3 == 0:
                    for i in range(size):
                        for j in range(size):
                            if j % 2 and j + 1 < size:
                                edge_list.append(((i, j), (i, j + 1)))
                            else:
                                continue
                    for l in range(size):
                        if switch % 2 and switch != 0 and l + 1 < size:
                            if l % 2:
                                edge_list.append(((l, 0), (l + 1, 0)))
                        else:
                            if l % 2 == 0 and l + 1 < size:
                                edge_list.append(((l, 0), (l + 1, 0)))
                elif layer % 4 == 0:
                    for i in range(size):
                        if i % 2 and i + 1 < size:
                            for j in range(size):
                                edge_list.append(((i, j), (i + 1, j)))
                        else:
                            continue
                    for l in range(size):
                        if switch % 2 and switch != 0 and l + 1 < size:
                            if l % 2:
                                edge_list.append(((0, l), (0, l + 1)))
                        else:
                            if l % 2 == 0 and l + 1 < size:
                                edge_list.append(((0, l), (0, l + 1)))
                else:
                    for i in range(size):
                        for j in range(size):
                            if j % 2 or j + 2 > size:
                                continue
                            else:
                                edge_list.append(((i, j), (i, j + 1)))
                    for l in range(size):
                        if switch % 2 and switch != 0 and l + 1 < size:
                            if l % 2:
                                edge_list.append(((l, size - 1), (l + 1, size - 1)))
                        else:
                            if l % 2 == 0 and l + 1 < size:
                                edge_list.append(((l, size - 1), (l + 1, size - 1)))
            else:
                if layer > 4 and layer % 4:
                    layer = layer % 4
                elif layer > 4:
                    layer = 4
                if layer % 2 == 0 and layer % 4 != 0:
                    for i in range(size):
                        if i % 2 or i + 2 > size:
                            continue
                        else:
                            for j in range(size):
                                edge_list.append(((i, j), (i + 1, j)))
                elif layer % 3 == 0:
                    for i in range(size):
                        for j in range(size):
                            if j % 2 and j + 1 < size:
                                edge_list.append(((i, j), (i, j + 1)))
                            else:
                                continue
                elif layer % 4 == 0:
                    for i in range(size):
                        if i % 2 and i + 1 < size:
                            for j in range(size):
                                edge_list.append(((i, j), (i + 1, j)))
                        else:
                            continue
                else:
                    for i in range(size):
                        for j in range(size):
                            if j % 2 or j + 2 > size:
                                continue
                            else:
                                edge_list.append(((i, j), (i, j + 1)))
            n = 0
            matrix_number = {}
            for i in range(size):
                for j in range(size):
                    matrix_number[(i, j)] = n
                    n += 1

            entang_list = [(matrix_number[i[0]], matrix_number[i[1]]) for i in edge_list]
        if self.connectivity == "random":
            import random
            qubits_list = list(range(self.size))
            random.shuffle(qubits_list)
            entang_list = [(qubits_list[i], qubits_list[i + 1]) for i in range(0, len(qubits_list) - 1, 2)]
        if 'regular' in self.connectivity:
            d = int(self.connectivity.replace('-regular', ''))
            p = math.floor(layer % self.size)
            seed = 0
            graph = nx.random_regular_graph(d, self.size, seed)
            while nx.is_connected(graph) is False:
                seed += 1
                graph = nx.random_regular_graph(d, self.size, seed)
            entang_list = []
            while len(graph) > 0:
                while p >= len(graph):
                    p -= 1
                if len(list(graph[list(graph.nodes())[p]].keys())) == 0:
                    p = len(graph)
                    while p > 0:
                        p -= 1
                        if len(list(graph[list(graph.nodes())[p]].keys())) == 0:
                            continue
                        else:
                            break
                    if p == 0:
                        break
                p_node = p % d
                while p_node >= len(list(graph[list(graph.nodes())[p]].keys())):
                    p_node -= 1
                entang_list.append((list(graph.nodes())[p], list(graph[list(graph.nodes())[p]].keys())[p_node]))
                graph.remove_nodes_from([list(graph[list(graph.nodes())[p]].keys())[p_node], list(graph.nodes())[p]])

            for i in entang_list:
                if i in self.entang_list or tuple(reversed(i)) in self.entang_list:
                    entang_list = self._define_connectivity_(layer + 2)
            self.entang_list = entang_list
        return entang_list

    def _qibo_circuit_(self, c):
        def SU4(c, q, p, i, ii, iii, iv, v, vi, vii, viii, ix):
            c.add(gates.U3(q, theta=i, phi=ii, lam=iii, trainable=True))
            c.add(gates.U3(p, theta=iv, phi=v, lam=vi, trainable=True))
            c.add(gates.RXX(q, p, theta=vii, trainable=True))
            c.add(gates.RYY(q, p, theta=viii, trainable=True))
            c.add(gates.RZZ(q, p, theta=ix, trainable=True))

        qibo_rotation_dictionary = {'RX': lambda q: gates.RX(q, theta=0, trainable=True),
                                    'RY': lambda q: gates.RY(q, theta=0, trainable=True),
                                    'RZ': lambda q: gates.RZ(q, theta=0, trainable=True),
                                    'U3': lambda q: gates.U3(q, theta=0, phi=0, lam=0, trainable=True),
                                    'U2': lambda q: gates.U2(q, phi=0, lam=0, trainable=True),
                                    'GPI': lambda q: gates.GPI(q, phi=0, trainable=True),
                                    'GPI2': lambda q: gates.GPI2(q, phi=0, trainable=True), }
        qibo_entang_dictionary = {'CZ': lambda q, p: gates.CZ(q, p), 'CNOT': lambda q, p: gates.CNOT(q, p),
                                  'MS': lambda q, p: gates.MS(q, p, phi0=0, phi1=0, theta=0, trainable=True),
                                  'RYY': lambda q, p: gates.RYY(q, p, theta=0, trainable=True),
                                  'RXX': lambda q, p: gates.RXX(q, p, theta=0, trainable=True),
                                  'RZZ': lambda q, p: gates.RZZ(q, p, theta=0, trainable=True),
                                  'CU3': lambda q, p: gates.CU3(q, p, theta=0, phi=0, lam=0, trainable=True),
                                  'CRZ': lambda q, p: gates.CRZ(q, p, theta=0, trainable=True),
                                  'CX': lambda q, p: gates.CRX(q, p, theta=math.pi, trainable=False),
                                  'CY': lambda q, p: gates.CRY(q, p, theta=math.pi, trainable=False),
                                  'SU4': lambda q, p: SU4(q, p, i=0, ii=0, iii=0, iv=0, v=0, vi=0, vii=0, viii=0, ix=0)}
        counter_taylor = 0

        for l in range(0, self.p):
            entang_list = self._define_connectivity_(l)
            rotation = self.rotation
            entanglement = self.entanglement
            if l % 2 == 0:
                if self.rotation == 'Taylor_efficient':
                    if counter_taylor % 3 == 1:
                        rotation = 'RX'
                    elif counter_taylor % 3 == 2:
                        rotation = 'RY'
                    elif counter_taylor % 3 == 0:
                        rotation = 'RZ'
                    counter_taylor += 1
                for q in range(self.size):
                    if q in [x for t in entang_list for x in t] or l == 0:
                        if self.rotation == 'IONQ_U2':
                            c.add(qibo_rotation_dictionary['GPI'](q))
                            c.add(qibo_rotation_dictionary['GPI2'](q))
                        if self.rotation == 'Quantinuum':
                            c.add(qibo_rotation_dictionary['U1q'](q))
                            c.add(qibo_rotation_dictionary['RZ'](q))
                        elif self.rotation == "None":
                            pass
                        else:
                            c.add(qibo_rotation_dictionary[rotation](q))
                    else:
                        continue
            else:
                if self.entanglement == 'Taylor_efficient':
                    if counter_taylor % 3 == 1:
                        entanglement = 'RXX'
                    elif counter_taylor % 3 == 2:
                        entanglement = 'RYY'
                    elif counter_taylor % 3 == 0:
                        entanglement = 'RZZ'
                    counter_taylor += 1
                for q in entang_list:
                    if None in q:
                        continue
                    else:
                        if self.entanglement == 'SU4':
                            SU4(c, q[0], q[1], i=0, ii=0, iii=0, iv=0, v=0, vi=0, vii=0, viii=0, ix=0)
                        else:
                            c.add(qibo_entang_dictionary[entanglement](q[0], q[1]))
        return c


    def _qiskit_circuit_(self, c):
        param = ParameterVector('p', 0)
        params_number = 0
        qiskit_rotation_dictionary = {'RX': lambda q,: RXGate(q), 'RY': lambda q: RYGate(q),
                                      'RZ': lambda q: PhaseGate(q), 'U3': lambda q, p, r: U3Gate(q, p, r)}
        qiskit_entang_dictionary = {'CZ': CZGate(), 'RXX': lambda q, p: RXXGate(q, p)}
        for l in range(0, self.p):
            entang_list = self._define_connectivity_(l)
            entang_list = [(abs(i[0] - self.size + 1), abs(i[1] - self.size + 1)) for i in entang_list]
            if l % 2 == 0:
                for q in range(self.size):
                    q = self.size - q - 1
                    if q in [x for t in entang_list for x in t] or l == 0:
                        if self.rotation == 'U3':
                            params_number += 1
                            param.resize(params_number)
                            c.append(qiskit_rotation_dictionary['RX'](param[len(param) - 1]), [q])
                            params_number += 1
                            param.resize(params_number)
                            c.append(qiskit_rotation_dictionary['RY'](param[len(param) - 1]), [q])
                        else:
                            params_number += 1
                            param.resize(params_number)
                            c.append(qiskit_rotation_dictionary[self.rotation](param[len(param) - 1]), [q])
                    else:
                        continue
            else:
                for q in entang_list:
                    if None in q:
                        continue
                    else:
                        if self.entanglement == 'RXX':
                            params_number += 1
                            param.resize(params_number)
                            c.append(qiskit_entang_dictionary[self.entanglement](param[len(param) - 1]), [q[0], q[1]])
                        else:
                            c.append(qiskit_entang_dictionary[self.entanglement], [q[0], q[1]])
        return c

    def rotate_x_circuit(self):
        circuit = self.circuit_representation.copy()
        if self.name_library == 'qibo':
            circuit.add(gates.H(q) for q in range(self.size))
        elif self.name_library == 'qiskit':
            for i in range(self.size):
                circuit.append(HGate(), [i])
        return circuit

    def rotate_y_circuit(self):
        circuit = self.circuit_representation.copy()
        if self.name_library == 'qibo':
            circuit.add(gates.S(q).dagger() for q in range(self.size))
            circuit.add(gates.H(q) for q in range(self.size))
        elif self.name_library == 'qiskit':
            for i in range(self.size):
                circuit.append(SdgGate(), [i])
                circuit.append(HGate(), [i])
        return circuit

    @staticmethod
    def rotate_x_measure(state, name_library, nqubits =None):
        nqubits_circuit = ceil(log2(len(state)))
        if nqubits is None:
            nqubits = nqubits_circuit

        if name_library == 'qibo':
            from qibo import gates, models
            circuit = models.Circuit(nqubits_circuit)
            circuit.add(gates.H(q) for q in range(nqubits))
            state = circuit.execute(state).state()
        elif name_library == 'qiskit':
            from qiskit import QuantumCircuit
            c = QuantumCircuit(nqubits_circuit)
            for i in range(nqubits):
                c.append(HGate(), [i])
            state = state.evolve(c)

        return state

    @staticmethod
    def rotate_y_measure(state, name_library, nqubits =None):

        nqubits_circuit = ceil(log2(len(state)))
        if nqubits is None:
            nqubits = nqubits_circuit


        if name_library == 'qibo':
            from qibo import gates, models
            circuit = models.Circuit(nqubits_circuit)
            circuit.add(gates.H(q).dagger() for q in range(nqubits))
            circuit.add(gates.S(q).dagger() for q in range(nqubits))
            circuit.add(gates.H(q) for q in range(nqubits))
            state = circuit.execute(state).state()

        elif name_library == 'qiskit':
            from qiskit import QuantumCircuit
            c = QuantumCircuit(nqubits_circuit)
            for i in range(nqubits):
                c.append(SdgGate(), [i])
                c.append(HGate(), [i])
            state = state.evolve(c)
        return state
