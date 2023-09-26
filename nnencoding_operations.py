import random
from encoding_operations import EncodingOperations
from assistant import Assistant
from nn_encoding import NNEncoding


## Child Class of EncodingOperations which contains overridden function for Symbolic Regression Type Encoding.
class NNEncodingOps(EncodingOperations):

    ## Constructor
    # @param blueprint Blueprint dictionary to know the rules of encoding.
    def __init__(self, blueprint):
        super().__init__(blueprint)

    ## Generates Encoding for Individual
    # @return **encoding**
    def generate_encoding(self):
        encoding = self.__generate_random_expression()
        return encoding

    ## Generate random mathematical expression based on the rules in the blueprint dictionary.
    # @param depth integer value for depth to keep track during recursion.
    # @return Encoding in form of a list of string which combines to make a math expression.
    def __generate_random_expression(self, depth=0):
        blueprint_data = self.get_blueprint()

        # num_input = blueprint_data['num_input']
        # num_output = blueprint_data['num_output']
        # activation_funcs = blueprint_data['activation_func']
        # weight_init = blueprint_data['weight_init']
        # fully_connected = blueprint_data['fully_connected']
        # initial_hidden_neurons = blueprint_data['initial_hidden_neurons']
        # bias = blueprint_data['bias']
        # initial_connection = blueprint_data['initial_connection']

        nn = NNEncoding(blueprint_data)

        return nn

    ## Counts nodes in given encoding / Counts nodes connected to given node.
    # @param node Starting node.
    def count_nodes(self, node):

        # Checks for the type given if it is a list that means more than one nodes exists.
        if not (type(node) == type(list())):
            return 1

        # Recursive calls
        a1 = self.count_nodes(node[1])
        a2 = self.count_nodes(node[2])

        # Return the sum of recursive calls plus 1 for the root.
        return a1 + a2 + 1

    ## Checks the depth of the give encoding / node.
    # @param node Starting node.
    # @param depth integer value for depth to keep track during recursion.
    # @return depth count.
    def depth(self, node, depth=0):

        # Checks for the type given if it is a list that means more than one nodes exists.
        if not (type(node) == type(list())):
            return depth

        depth += 1

        # Recursive calls
        d1 = self.depth(node[1], depth)
        d2 = self.depth(node[2], depth)

        # Check the depth returned by recursive calls and return the greater count
        return d1 if d1 > d2 else d2

    ## Select a random point / node from Encoding.
    # @param node Encoding.
    # @return node / point
    def choose_rand_node(self, node):

        # Get total number of connected nodes.
        node_count = self.count_nodes(node)

        # Get a random point or node
        node_to_explore = random.randint(0, node_count)

        # Return node
        return node_to_explore

    ##  Replaces the node / point in the Encoding with a replacement node(s).
    # @param node Given node / point to replace.
    # @param replacement Node to replace the given point with.
    # @param cur_node Recursive parameter to keep current node.
    # @return new encoding / node.
    def replace_node(self, node, replacement, node_num, cur_node=0):
        if cur_node == node_num:
            return [replacement, (cur_node + 1)]
        cur_node += 1
        if not (type(node) == type(list())):
            return [node, cur_node]
        a1, cur_node = self.replace_node(node[1], replacement, node_num, cur_node)
        a2, cur_node = self.replace_node(node[2], replacement, node_num, cur_node)
        return [[node[0], a1, a2], cur_node]

    ## Prunes / Cuts the Encoding from a given point / node.
    # @param node Given node / point to prune
    # @param depth Recursive parameter to keep depth
    # @return Pruned Encoding
    def prune(self, node, depth=0):
        terms = self.get_blueprint()['terms']
        return self.__prune(node, terms)

    ## Prunes / Cuts the Encoding from a given point / node (Private function specific for Symbolic Regression).
    # @param node Given node / point to prune.
    # @param terms Mathematical terms from blueprint.
    # @param depth Recursive parameter to keep depth.
    # @return Pruned Encoding
    def __prune(self, node, terms, depth=0):
        if depth == Assistant.MAX_DEPTH - 1:
            t = terms[random.randint(0, len(terms) - 1)]
            return random.uniform(-0.5, 0.5) if t == 'R' else t

        depth += 1

        if not (type(node) == type(list())):
            return node
        a1 = self.__prune(node[1], terms, depth)
        a2 = self.__prune(node[2], terms, depth)
        return [node[0], a1, a2]

    ## Serializes the encoding into a string format.
    # @param node Node / Encoding to be Serialized.
    # @return Serialized string form
    def serialize_encoding(self, node):
        if not (type(node) == type(list())):
            return node
        return "({1} {0} {2})".format(node[0], self.serialize_encoding(node[1]), self.serialize_encoding(node[2]))
