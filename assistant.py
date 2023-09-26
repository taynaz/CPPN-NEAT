import enum
import os
import json
import numpy as np
import math

from datetime import datetime


class Neuron_type(enum.Enum):
    input_neuron = 1
    hidden_neuron = 2
    output_neuron = 3


## A Utility class, that stores constants and misc functions
class Assistant:
    ## Number of Individuals in a generation
    N_START_INDIVIDUALS = 50

    ## Max number of generations
    MAX_GENERATIONS = 30

    ## Probability which determines if an Individual should be explored / Mutated.
    MUTATION_PROBABILITY = 0.05

    ## Probability which determines if two Individuals should be reformed. (NOT IN USE)
    CROSSOVER_PROBABILITY = 0.90

    ## Probability which determines if Individual(s) should reproduce.
    REPRODUCTION_PROBABILITY = 0.4

    ## Probability which determines how many Individuals in each generation will survive
    SURVIVAL_PROBABILITY = 0.4

    ## NOT IN USE
    FITNESS_THRESHOLD = 1000

    ## Maximum depth of Encoding
    MAX_DEPTH = 5

    ## Directory where reports are saved
    REPORT_DIR = 'evo_report' + '_' + datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    ## File where information about each generation is saved
    GENERATION_INFO_FILE = REPORT_DIR + '/' + 'gen_info.txt'

    ## File where information about Fitness of each Individual is saved
    FITNESS_INFO_FILE = REPORT_DIR + '/' + 'fitness_info.txt'

    ## File where information about Raw Fitness of each Individual is saved
    RAW_FITNESS_INFO_FILE = REPORT_DIR + '/' + 'raw_fitness_info.txt'

    ## File where information about the parent info of each Individual is saved
    ANCESTRAL_INFO_FILE = REPORT_DIR + '/' + 'ancestral_info.txt'

    ## Probability which determines if a new node should be added
    ADD_NODE_PROBABILITY = 0.2

    ## Probability which determines if an existing node should be removed
    REMOVE_NODE_PROBABILITY = 0.2

    @staticmethod
    def first(s):
        '''Return the first element from an ordered collection
        or an arbitrary element from an unordered collection.
        Raise StopIteration if the collection is empty.
        '''
        return next(iter(s.items()))

    ## Puts the string in the given file
    # @param serialized_objecct string form of the object to be put
    # @param file Path to file
    @staticmethod
    def to_file(serialized_object, file):
        if not os.path.exists(Assistant.REPORT_DIR):
            os.mkdir(Assistant.REPORT_DIR)

        with open(file, 'a') as f:
            f.write(serialized_object)

    ## Converts and entire generation's Individuals into string
    # @param population Poupulation Object
    # @param n_gen generation ID
    @staticmethod
    def serialize_generation(population, n_gen):
        inds = population.get_individuals(n_gen)
        sstring = 'Generation {0}:\n\t Number of individuals: {1}\n\t'.format(str(n_gen), str(len(inds)))
        for ind in inds:
            t_str = 'Individual {0}:\n\t parent_id: {1}\n\t group: {2}\n\t fitness: {3}\n\t encoding: {4}\n\t'.format(
                ind.id, ind.parents, ind.get_group(), ind.get_fitness(), ind.get_encoding())
            sstring += t_str
        return sstring + '\n'

    ## Converts all the parents of an Individual into string
    # @param population Poupulation Object
    # @param ind_id Individual ID
    @staticmethod
    def serialize_ancestor(population, ind_id):
        inds = population.individuals
        ind = inds[ind_id]
        sstring = 'Individual {0}:\n\t parent_id: {1}\n\t group: {2}\n\t fitness: {3}\n\t encoding: {4}\n\t'.format(
            ind.id, ind.parents, ind.get_group(), ind.get_fitness(), ind.get_encoding())

        while not (ind.parents == -1):
            ind = inds[ind.parents]
            t_str = 'Individual {0}:\n\t parent_id: {1}\n\t group: {2}\n\t fitness: {3}\n\t encoding: {4}\n\t'.format(
                ind.id, ind.parents, ind.get_group(), ind.get_fitness(), ind.get_encoding())
            sstring += t_str

        return sstring + '\n'

    ## Converts raw fitness values into string
    # @param ind Individual
    # @param results Results of Evaluation before fitness calculation
    @staticmethod
    def serialize_raw_fitness(ind, results):
        s_res = json.dumps(results)

        sstring = 'Individual {0}: \n\t Fitness: {1}\n\t Raw Evalutions: {2}\n\t'.format(ind.id, ind.get_fitness(),
                                                                                         s_res)
        return sstring + '\n'

    @staticmethod
    def read_json(filename):
        with open(filename, 'r') as blueprint:
            data = json.load(blueprint)
        return data

    @staticmethod
    def dict_converter(data):
        data_list_dict = {"all_node_data": data[0], "all_weights_list": data[1],
                          "node_to_link_data": data[2]}

        return data_list_dict

    @staticmethod
    def save_json(data_list_dict, filename):
        with open(filename + '.json', 'w') as f:
            json.dump(data_list_dict, f)

    @staticmethod
    def clamp_values(activation_data):

        activation_data = np.round(activation_data, 4)

        activation_data[activation_data.any() == np.nan or activation_data.any() == np.isinf] = 1

        activation_data = np.maximum(-30, np.minimum(30, activation_data))

        return activation_data
