from individual import Individual
from collections import defaultdict, OrderedDict
from copy import deepcopy


## It keeps the record of all the Individuals and Generations, along with some helper functions to retrieve and update them.
class Population:

    ## Constructor
    def __init__(self):

        ## (Dict) Stores all the Individuals as (key,value) (id, Individual) pair
        self.individuals = {}

        ## (Dict) Stores the ids of the individuals in a list with generation id as key.
        self.generations = defaultdict(list)

        ## (List) Keeps record of best individuals.
        self.best_individuals = []

    # TODO: Work on an update system for Individuals
    ## Updates the encoding of an Individual based on the id.
    # @param individual_id Id of the Individual.
    # @param update encoding to update with.
    def update_individual(self, individual_id, update):
        self.individuals[individual_id].set_encoding(update)

    ## Adds an Individual to the Population.
    # @param individual Individual Object.
    def add_individual(self, individual):
        self.individuals[individual.id] = individual
        self.generations[individual.generation].append(individual.id)

    ## Retrieve all the Individuals in a generation
    # @param generation Generation id to retireve
    # @return List of individuals
    def get_individuals(self, generation):
        gen_inds = self.generations[generation]
        inds = [self.individuals[i_id] for i_id in gen_inds]
        return inds

    ## Retrieves the parent of the given individual.
    # @param ind Individual object
    # @return parent of Individual (Individual Object)
    def get_parent(self, ind):
        return self.individuals[ind.parents]

        ## Sorts the Individuls by fitness.

    def sort_individuals_by_fitness(self):
        self.individuals = OrderedDict(sorted(self.individuals.items(), key=lambda x: x[1].get_fitness(), reverse=True))

    ## Retrieve a copy of sorted Individuals of a generation by fitness
    # @param generation ID of the generation
    # @return sorted Individuals
    def get_sorted_individuals_by_fitness(self, generation):

        # Make a deep copy of Individuals
        dc_inds = deepcopy(self.individuals)

        # Sort them
        srt_inds = OrderedDict(sorted(dc_inds.items(), key=lambda x: x[1].get_fitness(), reverse=True))

        # Get the Individuals from the generaion
        gen_inds = self.generations[generation]
        inds = [srt_inds[i_id] for i_id in gen_inds]

        # Return them
        return inds

    ## Get the sum of the fitnesses in the given generation
    # @param generation ID of the generation
    # @return sum of the fitnesses
    def get_sum_fitness(self, generation):

        # Get the Individual IDs
        gen_inds = self.generations[generation]
        sum_fitness = 0

        # Iterate through individuals and sum fitnesses
        for i_id in gen_inds:
            # LOOP START
            sum_fitness += self.individuals[i_id].get_fitness()
            # LOOP END

        return sum_fitness

    ## Get the N best Individuals
    # @param n Number of Individuals to retrieve
    # @return List of best Individuals
    def get_best(self, n):

        best = []

        # Get the iterator for individuals
        ind_iter = iter(self.individuals.items())
        for i in range(n):
            # LOOP START
            best.append(next(ind_iter))
        # LOOP END

        return best
