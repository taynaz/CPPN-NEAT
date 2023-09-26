from assistant import Neuron_type, Assistant
import math
import random
from path_finding import path_check
from feed_forward import feed_forward
import numpy as np
from weight_init import WeightInitializer
from activation import Activation
import time
import threading
from concurrent.futures import ProcessPoolExecutor as Pool
## Node is the single neuron in a neural network
class Node:
    id = 0

    ## constructor for Node Class. Node is a single neuron in the neural network
    # @params id int unique id for each neuron
    # @param neuron_type string specifying the neuron type; can be input, hidden, output, bias and none
    # @param activation_response is the result of activation function applied on the input coming to the node
    # @param splitY float value to determine y position in the 2d grid
    # @param splitX float value to determine x position in the 2d grid

    def __init__(self, neuron_type, activation_type, recurrent, activation_response, split_x, split_y):
        self.__nodeid = Node.id
        self.__neuron_type = neuron_type
        self.__recurrent = recurrent
        self.__activation_response = activation_response
        self.__split_x = split_x
        self.__split_y = split_y

        self.__activation_type = activation_type
        self.__active_status = True

        # Increment static counter.
        Node.id += 1

    def get_id(self):
        return self.__nodeid

    def get_neuron_type(self):
        return self.__neuron_type

    def get_recurrent(self):
        return self.__recurrent

    def get_activation_response(self):
        return self.__activation_response

    def get_activation_type(self):
        return self.__activation_type

    def get_split_y(self):
        return self.__split_y

    def get_split_x(self):
        return self.__split_x

    def get_active_status(self):
        return self.__active_status

    def set_id(self, id):
        self.__id = id

    def set_neuron_type(self, neuron_type):
        self.__neuron_type = neuron_type

    def set_recurrent(self, recurrent):
        self.__recurrent = recurrent

    def set_activation_response(self, activation_response):
        activation_response = Activation.computeAF(activation_response, self.__activation_type)(Activation)
        self.__activation_response = activation_response

    def set_split_y(self, split_y):
        self.__split_y = split_y

    def set_split_x(self, split_x):
        self.__split_x = split_x

    def set_active_status(self, status):
        self.__active_status = status

    def set_activation_type(self, activation_type):
        self.__activation_type = activation_type

    def serialize(self):
        node_list = [self.get_id(), self.get_neuron_type().value, self.get_activation_type()]

        return node_list


## Link is type of connection between two neurons
class Link:
    id = 0

    ## constructor for Link Class. Link is a single connection between two nodes in the neural network
    # @params in_neuron int unique id for neuron the link is coming into
    # @param  out_neuron int unique id for neuron the link is going out from
    # @param  weight double value referring to the weight of the link
    # @param  status bool value for representing if the link is ON or OFF
    # @param  recurrent bool value if the link is recurrent or not
    # @param  innovation_id int unique value to keep record of family history
    def __init__(self, in_neuron, out_neuron, weight, status, recurrent, innovation_id):
        self.__link_id = Link.id
        self.__in_neuron = in_neuron
        self.__out_neuron = out_neuron
        self.__weight = weight
        self.__status = status
        self.__recurrent = recurrent
        self.__innovation_id = innovation_id

        Link.id += 1
        return

    def get_id(self):
        return self.__link_id

    def get_in_neuron(self):
        return self.__in_neuron

    def get_out_neuron(self):
        return self.__out_neuron

    def get_weight(self):
        return self.__weight

    def get_status(self):
        return self.__status

    def get_recurrent(self):
        return self.__recurrent

    def get_innovation_id(self):
        return self.__innovation_id

    def serialize(self):
        link_list = [self.get_id(), self.get_in_neuron(), self.get_out_neuron(), self.get_weight()[0]]

        return link_list

    def set_in_neuron(self, in_neuron):
        self.__in_neuron = in_neuron

    def set_out_neuron(self, out_neuron):
        self.__out_neuron = out_neuron

    def set_weight(self, weight):
        weight = Assistant.clamp_values(weight)
        self.__weight = weight

    def set_status(self, status):
        self.__status = status

    def set_recurrent(self, recurrent):
        self.__recurrent = recurrent

    def set_innovation_id(self, innovation_id):
        self.__innovation_id = innovation_id


# NNEncoding contains all the nodes and links created in lists
class NNEncoding:
    # def __init__(self, num_input, num_output, activation_funcs, weight_init, fully_connected, initial_hidden_neurons,
    #              bias, initial_connection):
    def __init__(self, kwargs):

        self.nodes_list = []
        self.link_list = []
        self.output_result = []
        self.node_to_link_data = []
        self.num_output = 0
        self.num_input = 0
        self.output_index = []
        self.input_index = []
        self.weight_init = 0


        if ('filename' in kwargs.keys()):
            self.load_model(kwargs['filename'])

        else:
            self.init(kwargs['num_input'],
                      kwargs['num_output'],
                      kwargs['activation_func'],
                      kwargs['weight_init'],
                      kwargs['fully_connected'],
                      kwargs['initial_hidden_neurons'],
                      kwargs['bias'],
                      kwargs['initial_connection'])

    def init(self, num_input, num_output, activation_func, weight_init, fully_connected, initial_hidden_neurons,
             bias, initial_connection):
        self.num_input = num_input
        self.num_output = num_output
        self.activation_funcs = activation_func
        self.weight_init = weight_init
        self.fully_connected = fully_connected
        self.initial_hidden_neurons = initial_hidden_neurons
        self.bias = bias
        self.initial_connection = initial_connection


        max_index = self.num_input + self.num_output  # would decrease by 1 when appending to the list
        start_index = self.num_input
        self.output_index = [x for x in range(start_index, max_index)]
        self.input_index = [x for x in range(0, self.num_input)]

        self.__create_input_node()
        self.__create_output_node()
        self.__create_initial_hidden_neurons()
        self.__create_links()

    def __create_input_node(self):
        """ Creating the input nodes here """
        count = 0
        for i in range(0, self.num_input):
            count = count + 1
            self.nodes_list.append(Node(Neuron_type.input_neuron, 'identity', False, np.zeros([1]), count, 0))

    def __create_output_node(self):
        """ Creating the output nodes here """
        count = 0
        for i in range(0, self.num_output):
            count = count + 1
            self.nodes_list.append(Node(Neuron_type.output_neuron, 'sigmoid', False, np.zeros([1]), count, 0))

    def __create_initial_hidden_neurons(self):
        """ Creating the initial hidden nodes here """

        count = 0
        for i in range(0, self.initial_hidden_neurons):
            count = count + 1
            self.nodes_list.append(
                Node(Neuron_type.hidden_neuron, random.choice(self.activation_funcs), False, np.zeros([1]), count, 0))

    def __create_links(self):
        """" Imagine a 2d matrix with node ids on the axis. The matrix data will be the link assigned ids.
        But first we need to compute how many links can be created ( max & min). we ASSUME:
        *- The output node isn't interconnected to itself and other output nodes.
        Under this assumption the formula for maximum number of links is:
        [ all_nodes^2 - (num_input_nodes^2 + num_output_nodes^2)"""

        max_links_to_create = (self.num_input * (self.initial_hidden_neurons + self.num_output)) + \
                              (self.num_output * self.initial_hidden_neurons) + self.initial_hidden_neurons
        min_links_to_create = max(self.num_input, self.num_output)

        links_to_create = random.randint(min_links_to_create, max_links_to_create)

        """Initialize 2d matrix with node id's as the (x,y) indices and data to be filled with weight id's"""
        rows, cols = (len(self.nodes_list), len(self.nodes_list))
        self.node_to_link_data = [[-1 for i in range(cols)] for j in range(rows)]

        """We assume the creation of nodes in the order of: input, output and hidden is untouched. By this assumption 
        the first node id's will be equivalent to the how many input nodes we have. Same for output. The hidden nodes
        are expected to change throughout the evolution"""

        """ Assigning -2 to input node positions as these cant be interconnected hence link id exists between them"""

        for y in self.input_index:
            for x in range(0, len(self.node_to_link_data)):
                self.node_to_link_data[x][y] = -2

        """Assigning -2 to output node position as a output neuron cannot create a link to itself.
         This can be CHANGED IN THE FUTURE"""

        """First find all the output indexes and append to the list"""

        for index in self.output_index:
            for x in range(0, len(self.node_to_link_data)):
                self.node_to_link_data[index][x] = -2

        self.link_list = []
        """First priority is to create at least one connection from input to output nodes """
        output_index_iterator = random.randint(0, len(self.output_index) - 1)

        for input_index in self.input_index:


            link_weight =  WeightInitializer.get_weight(self.weight_init)(WeightInitializer)
            new_link = Link(input_index, self.output_index[output_index_iterator], link_weight, True, 0, 0)
            self.node_to_link_data[input_index][self.output_index[output_index_iterator]] = (new_link.get_id())
            self.link_list.append(new_link)
            output_index_iterator = output_index_iterator + 1
            links_to_create = links_to_create - 1
            if output_index_iterator > len(self.output_index) - 1:
                output_index_iterator = 0

        while links_to_create > 0:
            rand_x = random.randint(0, len(self.node_to_link_data) - 1)
            rand_y = random.randint(0, len(self.node_to_link_data) - 1)
            link_weight = WeightInitializer.get_weight(self.weight_init)(WeightInitializer)
            if (rand_x == rand_y) or (self.node_to_link_data[rand_x][rand_y] > -1):
                continue
            elif (self.node_to_link_data[rand_x][rand_y] > -2) and (self.node_to_link_data[rand_y][rand_x] <= -1):
                new_link = Link(rand_x, rand_y, link_weight, True, 0, 0)
                self.link_list.append(new_link)
                self.node_to_link_data[rand_x][rand_y] = (new_link.get_id())  # insert link id's

                links_to_create = links_to_create - 1


    """TESTING FOR NOW"""
    def assign_input_activation(self, in_loc, input_vals):
        self.nodes_list[in_loc].set_activation_response(input_vals)

    def assign_input_activation_multiProcessing(self, in_loc):
        self.nodes_list[in_loc].set_activation_response(self.input_array[in_loc])


    def get_output_result(self, inputs_array):

        for in_loc, input_vals in zip(self.input_index, inputs_array):
            self.nodes_list[in_loc].set_activation_response(input_vals)

        start_feed = time.time()
        feed_forward(self.node_to_link_data, self.link_list, self.nodes_list)
        end_feed = time.time()

        for out_in in self.output_index:
            self.output_result.append(self.nodes_list[out_in].get_activation_response())

        return self.output_result


    def save_model(self, filename):
        print('saving model')
        all_node_list = []
        for nodes in self.nodes_list:
            all_node_list.append(nodes.serialize())

        all_weights_list = []
        for link in self.link_list:
            all_weights_list.append(link.serialize())

        Assistant.save_json(
            Assistant.dict_converter(
                [all_node_list, all_weights_list, self.node_to_link_data]), filename)

    def load_model(self, path):
        print('loading model from' + path)
        Link.id = 0
        Node.id = 0

        model_data = Assistant.read_json(path)

        i = 0
        for node_data in model_data["all_node_data"]:
            if node_data[1] == 1:  # input neuron
                self.nodes_list.append(Node(Neuron_type.input_neuron, node_data[2], False, np.zeros([1]), 0, 0))
                self.num_input = self.num_input + 1
            if node_data[1] == 3:  # output neuron
                self.nodes_list.append(Node(Neuron_type.output_neuron, node_data[2], False, np.zeros([1]), 0, 0))
                # self.output_index.append(node_data[0])
                self.num_output = self.num_output + 1
            if node_data[1] == 2:  # hidden neuron
                self.nodes_list.append(Node(Neuron_type.hidden_neuron, node_data[2], False, np.zeros([1]), 0, 0))

            i = i + 1

        max_index = self.num_input + self.num_output  # would decrease by 1 when appending to the list
        start_index = self.num_input

        self.output_index = [x for x in range(start_index, max_index)]
        self.input_index = [x for x in range(0, self.num_input)]

        for link in model_data["all_weights_list"]:
            weight = np.array([link[3]])
            new_link = Link(link[1], link[2], weight, True, 0, 0)
            self.link_list.append(new_link)

        self.node_to_link_data = model_data["node_to_link_data"]

    def add_node(self, activation_list, weight_init_list):

        """Creating New Hidden Neuron"""
        self.nodes_list.append(
            Node(
                Neuron_type.hidden_neuron,
                random.choice(activation_list),
                False,
                np.zeros([1]),
                0,
                0
            )
        )

        """Adding another column to accommodate the new hidden node created"""
        for row in self.node_to_link_data:
            lastcolumn = -1
            # Now add the new column to the current row
            row.append(lastcolumn)
        """Create an array equivalent of number of nodes present: Useful for more than 1 addition of Node"""
        new_row = [-1] * len(self.nodes_list)

        """Add another row at the bottom to accommodate the new hidden created"""
        self.node_to_link_data.append(new_row)

        # id of node added
        new_node_id = (self.nodes_list[-1]).get_id()

        for y in self.input_index:
            self.node_to_link_data[new_node_id][y] = -2

        for x in self.output_index:
            self.node_to_link_data[x][new_node_id] = -2

        link_added = False
        while not link_added:
            rand_y = random.randint(0, len(self.node_to_link_data) - 1)
            link_weight = WeightInitializer.get_weight(weight_init_list)(WeightInitializer)
            if (new_node_id == rand_y) or (self.node_to_link_data[new_node_id][rand_y] > -1):
                continue
            elif (self.node_to_link_data[new_node_id][rand_y] > -2) and (
                    self.node_to_link_data[rand_y][new_node_id] <= -1):

                c1 = random.choice(self.output_index)
                c2 = random.choice(self.input_index)
                new_link = Link(new_node_id, c1, link_weight, True, 0, 0)
                new_link_test = Link(c2, new_node_id, WeightInitializer.get_weight(weight_init_list)(WeightInitializer),
                                     0, 0, 0)
                self.node_to_link_data[new_node_id][c1] = (new_link.get_id())  # insert link id's
                self.node_to_link_data[c2][new_node_id] = (new_link_test.get_id())  # insert link id's

                self.link_list.append(new_link)
                self.link_list.append(new_link_test)
                link_added = True

    def remove_node(self):

        node_removed = False
        while not node_removed:
            remove_node_id = random.randint(0, len(self.node_to_link_data) - 1)
            if remove_node_id in self.input_index or remove_node_id in self.output_index:
                continue
            else:
                self.nodes_list[remove_node_id].set_active_status(False)
                node_removed = True

        for row in range(0, len(self.node_to_link_data)):
            for col in range(0, len(self.node_to_link_data)):
                if row == remove_node_id or col == remove_node_id:
                    self.node_to_link_data[row][col] = -3
                    self.node_to_link_data[col][row] = -3

    def change_node_activation(self, activation_list):
        active_nodes = 0
        for node in self.nodes_list:
            if node.get_active_status():
                active_nodes = active_nodes + 1

        node_actv_to_change = random.randint(1, active_nodes - 1)

        node_ids_changed_list = []
        while node_actv_to_change > 0:
            # pick link id of active link
            node_id_to_change = random.randint(0, len(self.nodes_list) - 1)
            if self.nodes_list[node_id_to_change].get_active_status() and self.nodes_list[
                node_id_to_change].get_id() not in node_ids_changed_list:
                # only then change
                new_activation = random.choice(activation_list)
                self.nodes_list[node_id_to_change].set_activation_type(
                    new_activation)
                node_ids_changed_list.append(self.link_list[node_id_to_change].get_id())

                node_actv_to_change = node_actv_to_change - 1


    def add_link(self, weight_init_list):
        link_added = False
        while not link_added:
            rand_in = random.randint(0, len(self.node_to_link_data) - 1)
            rand_out = random.randint(0, len(self.node_to_link_data) - 1)
            if rand_in in self.output_index or rand_out in self.input_index or self.node_to_link_data[rand_in][
                rand_out] > -1:
                continue
            elif (self.node_to_link_data[rand_in][rand_out] > -2) and (
                    self.node_to_link_data[rand_out][rand_in] > -2):

                link_weight = WeightInitializer.get_weight(weight_init_list)(WeightInitializer)
                new_link = Link(rand_in, rand_out, link_weight, True, 0, 0)
                self.link_list.append(new_link)
                self.node_to_link_data[rand_in][rand_out] = (new_link.get_id())
                link_added = True


    def remove_link(self):
        link_removed = False

        while not link_removed:
            remove_link_to_node = random.randint(0, len(self.node_to_link_data) - 1)
            remove_link_from_node = random.randint(0, len(self.node_to_link_data) - 1)
            if remove_link_from_node in self.output_index or remove_link_to_node in self.input_index:
                continue
            elif self.node_to_link_data[remove_link_from_node][remove_link_to_node] > -1:
                selected_link_id = self.node_to_link_data[remove_link_from_node][remove_link_to_node]
                self.node_to_link_data[remove_link_from_node][remove_link_to_node] = -1
                self.link_list[selected_link_id].set_status(False)
                link_removed = True


    def perturb_weight_value(self):
        active_links = 0
        for link in self.link_list:
            if link.get_status():
                active_links = active_links + 1

        links_to_change = random.randint(1, active_links - 1)

        link_ids_changed_list = []
        while links_to_change > 0:
            # pick link id of active link
            link_to_change = random.randint(0, len(self.link_list) - 1)
            action_list = ['add', 'sub']
            if self.link_list[link_to_change].get_status() and self.link_list[
                link_to_change].get_id() not in link_ids_changed_list:
                # only then change
                new_weight_adjustment = random.random()
                action = random.choice(action_list)
                if action == 'sub':
                    self.link_list[link_to_change].set_weight(
                        self.link_list[link_to_change].get_weight() - new_weight_adjustment)
                elif action == 'add':
                    self.link_list[link_to_change].set_weight(
                        self.link_list[link_to_change].get_weight() + new_weight_adjustment)
                link_ids_changed_list.append(self.link_list[link_to_change].get_id())

                links_to_change = links_to_change - 1

    def replace_weight_value(self, weight_init_list):
        active_links = 0
        for link in self.link_list:
            if link.get_status():
                active_links = active_links + 1

        links_to_change = random.randint(1, active_links - 1)

        link_ids_changed_list = []
        while links_to_change > 0:
            # pick link id of active link
            link_to_change = random.randint(0, len(self.link_list) - 1)
            if self.link_list[link_to_change].get_status() and self.link_list[
                link_to_change].get_id() not in link_ids_changed_list:
                # only then change
                new_weight = WeightInitializer.get_weight(weight_init_list)(WeightInitializer)

                self.link_list[link_to_change].set_weight(
                    new_weight)
                link_ids_changed_list.append(self.link_list[link_to_change].get_id())

                links_to_change = links_to_change - 1
        return
