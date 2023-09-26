from collections import defaultdict
import numpy as np
import time
from assistant import Assistant


class Feedforward:
    def __init__(self):
        self.connected_nodes_dict = defaultdict(list)

    def addEdge(self, u, v):
        self.connected_nodes_dict[u].append(v)

    def matrix_to_dict(self, node_to_link_data):
        for i in range(0, len(node_to_link_data)):
            for j in range(0, len(node_to_link_data)):
                if node_to_link_data[i][j] == -1 or node_to_link_data[i][j] == -2 or node_to_link_data[i][j] == -3:
                    pass
                else:
                    self.addEdge(i, j)

    def compute_network(self, node_to_link_data, links_data, node_data):
        activation_response = [0]

        for from_nodes in self.connected_nodes_dict.keys():

            if node_data[from_nodes].get_active_status():

                for to_node in self.connected_nodes_dict[from_nodes]:

                    weight_array = links_data[node_to_link_data[from_nodes][to_node]].get_weight()
                    activation_response_array = Assistant.clamp_values(node_data[from_nodes].get_activation_response())
                    activation_response = (np.multiply(activation_response_array, weight_array))
                    new_activation = (np.add(node_data[to_node].get_activation_response(), activation_response))
                    node_data[to_node].set_activation_response(new_activation)


def feed_forward(node_to_link_data, links_data, node_data):
    f = Feedforward()
    f.matrix_to_dict(node_to_link_data)
    f.compute_network(node_to_link_data, links_data, node_data)
    return
