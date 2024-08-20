# ==============================================================================
# Copyright 2023-* Marco Sciorilli. All Rights Reserved.
# Copyright 2023-* QRC @ Technology Innovation Institute of Abu Dhabi. All Rights Reserved.
# ==============================================================================


from typing import Tuple
import math
import MQLib
import networkx as nx
import numpy as np
from time import time
from .datamanager import connect_database,create_table


def _get_exact_solution( problem=None):
    np.random.seed(int(time()))
    instance = MQLib.Instance('M', nx.to_numpy_array(problem))
    def cb_fun(spins):
        return 1
    result = MQLib.runHeuristic('BURER2002', instance, 10, cb_fun)
    max_energy = result['objval']
    result_exact = max_energy
    return result_exact, result['solution']


def initialize_database(name_database: str) -> None:
    """
    Function that initialize the database for the benchmark, with a user defined name.
    :param name_database: Name of the database
    """
    rows = {'kind': 'TEXT', 'method': 'TEXT', 'instance': 'TEXT', 'trial': 'INT', 'layer_number': 'INT',
            'spins_number': 'INT',
            'optimization': 'TEXT', 'compression': 'FLOAT', 'pauli_string_length': 'INT',
            'graph_kind': 'TEXT', 'activation_function': 'TEXT', 'qubits': 'INT', 'solution_raw': 'TEXT',
            'solution_local': 'TEXT', 'shots':'INT',
            'unrounded_solution': 'TEXT', 'max_energy_raw': 'FLOAT', 'energy_ratio_raw': 'FLOAT',
            'max_energy_local': 'FLOAT', 'energy_ratio_local': 'FLOAT',
            'initial_parameters': 'TEXT', 'parameters': 'TEXT', 'number_parameters': 'INT',
            'hyperparameter': 'TEXT', 'epochs': 'INT',
            'time': 'FLOAT', 'loss_ratio': 'FLOAT', 'entanglement': 'TEXT', 'rotation': 'TEXT', 'connectivity': 'TEXT',
            'loss_name': 'TEXT', 'loss_value': 'FLOAT'}
    unique = ['kind', 'instance', 'layer_number', 'spins_number', 'optimization', 'compression', 'hyperparameter',
              'pauli_string_length',
              'connectivity', 'graph_kind', 'trial', 'activation_function', 'loss_name', 'entanglement', 'connectivity',
              'rotation', 'epochs', 'shots']
    connect_database(name_database)
    create_table(name_database, name_database, rows, unique)


def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    discriminant = b ** 2 - 4 * a * c
    if discriminant >= 0:
        x_1 = (-b + math.sqrt(discriminant)) / 2 * a
        x_2 = (-b - math.sqrt(discriminant)) / 2 * a
    else:
        x_1 = complex((-b / (2 * a)), math.sqrt(-discriminant) / (2 * a))
        x_2 = complex((-b / (2 * a)), -math.sqrt(-discriminant) / (2 * a))
    return x_1, x_2


def _round(num: float) -> int:
    """
    Rounding function
    :type num: float number
    """
    np.random.seed(0)
    if num > 0:
        return +1
    elif num < 0:
        return -1
    else:
        return np.random.choice([-1, 1], 1)[0]


def _value_edge(x1, x2, edge):
    return edge * (1 - x1 * x2) / 2

def local_search(solution_raw,  adjacency_matrix=None):
    solution = np.array([_round(i) for i in solution_raw])
    new_solution = _two_flip(adjacency_matrix, _one_flip(adjacency_matrix, solution)[1])[1]
    while not (new_solution == solution).all():
        solution = new_solution
        new_solution = _two_flip(adjacency_matrix, _one_flip(adjacency_matrix, new_solution)[1])[1]
    return _two_flip(adjacency_matrix, _one_flip(adjacency_matrix, solution)[1])


def _obj_value(node_mapping_expectation,  adjacency_matrix=None):
    return _cut_value(node_mapping_expectation, adjacency_matrix)

def _cut_value(node_mapping_expectation, adjacency_matrix):
    cut_value = 0
    for i in adjacency_matrix:
        cut_value += _value_edge(node_mapping_expectation[i[0]], node_mapping_expectation[i[1]],
                                 adjacency_matrix[i])
    return cut_value


def _update_cut(edges_dict, cut_to_update, adjacency_matrix):
    for j in edges_dict:
        cut_to_update -= _value_edge(edges_dict[j][0], edges_dict[j][1], adjacency_matrix[j])
        cut_to_update += _value_edge(-edges_dict[j][0], edges_dict[j][1], adjacency_matrix[j])
    return cut_to_update

def _update_cut_double(edges_dict, cut_to_update, adjacency_matrix, special):
    for j in edges_dict:
        if j == special:
            cut_to_update -= _value_edge(edges_dict[j][0], edges_dict[j][1], adjacency_matrix[j])
            cut_to_update += _value_edge(-edges_dict[j][0],-edges_dict[j][1], adjacency_matrix[j])
        else:
            cut_to_update -= _value_edge(edges_dict[j][0], edges_dict[j][1], adjacency_matrix[j])
            cut_to_update += _value_edge(-edges_dict[j][0], edges_dict[j][1], adjacency_matrix[j])
    return cut_to_update

def get_other_element(my_tuple, given_element):
    if my_tuple[0] == given_element:
        return my_tuple[1]
    elif my_tuple[1] == given_element:
        return my_tuple[0]


def _one_flip(adj_matrix, starting_solution):
    starting_cut = _cut_value(starting_solution, adj_matrix)
    best_cut = starting_cut
    best_solution = starting_solution.copy()
    for i in range(len(starting_solution)+1):
        temporary_solution = best_solution.copy()
        edges_list = {j: (temporary_solution[i], temporary_solution[get_other_element(j, i)]) for j in
                      adj_matrix.keys() if i in j}
        new_cut = _update_cut(edges_list, best_cut, adj_matrix)
        if new_cut > best_cut:
            temporary_solution[i] = -1 * temporary_solution[i]
            best_solution = temporary_solution.copy()
            best_cut = new_cut
    return best_cut, best_solution

def _two_flip(adj_matrix, starting_solution):
    starting_cut = _cut_value(starting_solution, adj_matrix)
    best_cut = starting_cut
    best_solution = starting_solution.copy()
    for i in adj_matrix.keys():
        temporary_solution = best_solution.copy()
        edges_list = {j: (temporary_solution[i[0]], temporary_solution[get_other_element(j, i[0])]) for j in
                      adj_matrix.keys() if i[0] in j}
        for j in adj_matrix.keys():
            if i[1] in j:
                edges_list[j] = (temporary_solution[i[1]], temporary_solution[get_other_element(j, i[1])])
        new_cut = _update_cut_double(edges_list, best_cut, adj_matrix,i)

        if new_cut > best_cut:
            temporary_solution[i[0]] = -1 * temporary_solution[i[0]]
            temporary_solution[i[1]] = -1 * temporary_solution[i[1]]
            best_solution = temporary_solution.copy()
            best_cut = new_cut

    return best_cut, best_solution


