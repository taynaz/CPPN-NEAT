from assistant import Assistant

## Individual containing the encoding
class Individual:
    ## A static ID counter
    __id = 0

    ## Constructor
    # @param encoding Encoding of the Individual.
    # @param parents Parent ID (its called parents because it may contain more than one IDs)
    # @param generation Generation ID to which the Individual belongs to
    def __init__(self, encoding, parents=-1, generation=0):
        # private variables

        ## fitness of the individual.
        self.__fitness = 0

        ## group the individual is associated with.
        self.__group = 0

        ## generation of the individual.
        self.generation = generation

        # public variables

        ## ID of the individual assigned through static counter.
        self.id = Individual.__id

        # Increment static counter.
        Individual.__id += 1

        ## Parents of the Individual
        self.parents = parents

        ## encoding of the Individual
        self.__encoding = encoding

    ## Retrieve the fitness
    # @return fitness (float)
    def get_fitness(self):
        return self.__fitness

    ## Set the fitness
    # @param fitness float value of score of the Individual after Evaluation
    def set_fitness(self, fitness):
        self.__fitness = fitness

    ## Retrieve the group ID
    # @return group (int)
    def get_group(self):
        return self.__group

    ## Set the group
    # @param group int value of ID of the Group the Individual is associated with.
    def set_group(self, group):
        self.__group = group

    ## Retrieve the encoding
    # @return encoding
    def get_encoding(self):
        return self.__encoding

    ## Set encoding
    # @param encoding of the individual
    def set_encoding(self, encoding):
        self.__encoding = encoding