from assistant import Assistant

import numpy as np

"""
    This Class still needs a lot of work, so please ignore the design choices for now.
"""


## Saves and Calculates the fitness of the Individuals
class Evaluator:

    ## Constructor
    def __init__(self):

        ## (Dict) Record of the all the raw scores from which the fitness is calculated of the Individuals saved as (key: id, value: List).
        self.__eval_record = {}

    ## Adds one record of the Individual
    # @param record A Dict format single entry
    def add_record(self, record):
        self.__eval_record.update(record)

    ## Retrieve the full record stored in Evaluator
    # @return (Dict) record
    def get_record(self):
        return self.__eval_record

    ## Calculates the fitness of the given individual.
    # @param individual Individual Object
    # @param results Dict form of evaluation
    def evaluate(self, individual, results):

        res = []
        for k in results.keys():
            try:
                res_lst = results[k]['PAWN']
                res_arr = np.array(res_lst)
                res.append(res_arr.sum())
            except:
                continue

        res = np.array(res)
        individual.set_fitness(res.sum())

        s_r_f = Assistant.serialize_raw_fitness(individual, results)
        Assistant.to_file(s_r_f, Assistant.FITNESS_INFO_FILE)
