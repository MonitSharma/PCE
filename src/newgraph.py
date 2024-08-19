from copy import deepcopy
from typing import Optional, Tuple, List
import networkx as nx
import numpy as np
import subprocess


class RandomGraphs(object):
    def __init__(self, index: int, nodes_number: int, true_random: Optional[bool] = False, fully_connected = False):
        self.index = index
        self.nodes_number = nodes_number
        self.fully_connected = fully_connected
        if true_random:
            np.random.seed(np.randint(0))
            self.index = np.randint(0)
        self.graph = self.create_graph()

    @staticmethod
    def complete_graph_instantiater(number: int = 50, size: int = 10) -> List:
        """
        Function which create a list of fully connected graphs with random weights.
        :param number: Number of graphs in the list
        :param size: Number of nodes in each graph
        :return: List of nx graphs
        """
        dummy_graph = nx.complete_graph(size)
        return [RandomGraphs.weighted_graph(graph=dummy_graph) in range(number)]

    @staticmethod
    def weighted_graph(graph: nx.Graph, weight_range: Optional[Tuple[float, float]] = (-10, 10),
                       integer_weights: Optional[bool] = False,
                       seed: Optional[int] = None) -> nx.Graph:
        """
        Takes an unweighted input graph and returns a weighted graph where the weights are uniformly sampled at random
        :param graph: The graph to weight
        :param weight_range: The range of values for the weights
        :param integer_weights: True if only integer weight are considered
        :param seed: Seed for the generation of the random weight
        :return: The weighted graph
        """
        # If seed is given, fix the seed
        if seed:
            np.random.seed(seed)
        # Do a deepcopy of the graph
        weighted_graph = deepcopy(graph)
        # Per each edge, assign a weight
        for edge in weighted_graph.edges:
            # weighted_graph[edge[0]][edge[1]]['weight'] = np.random.choice([-1, 1])
            if integer_weights:
                weighted_graph[edge[0]][edge[1]]['weight'] = np.random.randint(int(weight_range[0]),int(weight_range[1]))
            else:
                weighted_graph[edge[0]][edge[1]]['weight'] = np.random.uniform(weight_range[0], weight_range[1])
        return weighted_graph

    @staticmethod
    def softmax_1d(array: np.array) -> np.array:
        """
        Perform a softmax transformation of a numpy array
        :param array: the numpy array
        :return: the softmax numpy array
        """
        y = np.exp(array - np.max(array))
        f_x = y / np.sum(np.exp(array))
        return f_x

    @staticmethod
    def gnp_random_connected_graph(nodes_number: int, prob: float, seed: int) -> nx.Graph:
        """
        Generates a random undirected graph, similarly to an Erdős-Rényi
        graph, but enforcing that the resulting graph is conneted.
        :param nodes_number:Number of nodes
        :param prob: Probability of each edge to exist
        :param seed:
        :return: A connected graph
        """

        minimum_degree = 0
        while minimum_degree <= 1:
            graph = subprocess.run(["./rudy", "-rnd_graph", f"{nodes_number}", f"{int(prob * 100)}", f"{seed}"],
                                   capture_output=True)
            new = graph.stdout.decode("utf-8").splitlines()
            new.pop(0)
            new_file = []
            g = nx.Graph()
            for line in new:
                num_string = list(map(int, line.split(' ')))
                num_string[0] -= 1
                num_string[1] -= 1
                g.add_edge(num_string[0], num_string[1], weight=num_string[2])
                new_file.append(str(num_string[0]) + " " + str(num_string[1]) + " " + str(num_string[2]))
            minimum_degree = min([len(g.adj[i]) for i in g.adj])
            prob += 0.001
        return g

    def create_graph(self) -> nx.Graph:
        """
        Function which create a random connected graph.
        :return: A graph
        """
        # Initialise the probability of each node of having an edge with a neighbour
        if self.fully_connected:
            p = 1
        else:
            # Fix the probability if the same seed is used
            np.random.seed(self.index)

            def dens(nodes):
                if nodes>1700:
                    return 0.01
                else:
                    return np.exp(-0.8541 - 0.0055 * nodes + 4.003e-06 * nodes ** 2 - 1.158e-09 * nodes ** 3)
            p = np.random.uniform(6/(self.nodes_number-1),dens(self.nodes_number))
        # Create the graph unweighted
        graph = self.gnp_random_connected_graph(self.nodes_number, p, self.index)
        # Assign weight to the edges
        # graph = self.weighted_graph(graph, seed=0, softmax=self.softmax)
        return graph

    def return_index(self):
        return self.index

    @staticmethod
    def _graph_to_dict(graph):
        adj_matrix = nx.to_numpy_array(graph)
        edges = {}
        for i in range(adj_matrix.shape[0]):
            for j in range(i):
                if adj_matrix[i][j] == 0:
                    continue
                edges[(i, j)] = adj_matrix[i][j]
        return edges
