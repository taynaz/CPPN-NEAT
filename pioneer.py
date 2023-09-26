from assistant import Assistant, Neuron_type
from encoding_operations import EncodingOperations
from nn_encoding import Node
import numpy as np
import random


# # Is responsible to mutate Individuals in a given population, in other words its a pioneer to search for different
# possibilities in and encoding.
class Pioneer:

    ## Constructor
    # @param enc_ops An EncodingOperations object, required for performing certain operations
    def __init__(self, enc_ops):

        ## (EncodingOperations) required for performing certain operations
        self.encoding_ops = enc_ops

    ## Explores random possibilities in the generation of the given population
    # @param population Population object containing all the generations
    # @param generation Generation to explore
    def explore(self, population, generation):

        # Get the individuals in the given generation
        inds = population.get_individuals(generation)

        # Explore them
        for ind in inds:
            # LOOP START

            # Randomly determine to mutate based on MUTATION_PROBABILITY
            if random.uniform(0, 1) < Assistant.MUTATION_PROBABILITY:
                # Get the parent of the Individual
                parent = population.get_parent(ind)

                # Explore current Individual
                new_enc = self.__explore_one(parent, ind)

                # Update Encoding of the current Individual
                population.update_individual(ind.id, new_enc)

                # Utility functions to help serialize parents and put them in designated file.
                s_p_ind = Assistant.serialize_ancestor(population, ind.id)
                Assistant.to_file(s_p_ind, Assistant.ANCESTRAL_INFO_FILE)
        # LOOP END

    ## Explores random possibilities of the given child based on the parents encoding.
    # @param parent Parent of the child, The Encoding of the parent is explored and and given to the Child.
    # @param child Child to be updated.
    # @return new child encoding
    def __explore_one(self, parent, child):

        # Get encoding of the parent
        parent_enc = parent.get_encoding()

        # Choose random point to explore
        node_to_explore = self.encoding_ops.choose_rand_node(parent_enc)

        # Get encoding of the child
        child_enc = child.get_encoding()

        # Replace the child encoding with the point of the parent's Encoding
        new_child_enc, _ = self.encoding_ops.replace_node(parent_enc, child_enc, node_to_explore)

        # Prune to keep the MAX_DEPTH.
        new_child_enc = self.encoding_ops.prune(new_child_enc)

        # Return new Encoding
        return new_child_enc


class NNPioneer(Pioneer):

    def __init__(self, enc_ops):
        super().__init__(enc_ops)

    def node_add_mutation(self, nn_parent, nn_child):

        prob = random.uniform(0, 1)

        if prob < 0.1:
            nn_child.add_node(self.encoding_ops.get_blueprint()['activation_func'],
                              self.encoding_ops.get_blueprint()['weight_init'])
        if 0.1 < prob < 0.2:
            nn_child.remove_node()
        if 0.2 < prob < 0.4:
            nn_child.add_link(self.encoding_ops.get_blueprint()['weight_init'])
        if 0.4 < prob < 0.6:
            nn_child.remove_link()
        if 0.6 < prob < 0.7:
            nn_child.perturb_weight_value()
        if 0.7 < prob < 0.8:
            nn_child.replace_weight_value(self.encoding_ops.get_blueprint()['weight_init'])
        if 0.8 < prob < 1:
            nn_child.change_node_activation(self.encoding_ops.get_blueprint()['activation_func'])

        return nn_child

    def __link_mutation(self, node_list, link_list, node_to_link_data):
        return

    def __weight_mutation(self, node_list, link_list, node_to_link_data):
        return
