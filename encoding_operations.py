import random
import enum
# from nnencoding_operations import NNEncodingOps

'for non pycharm environment: from ImageEvolution.assistant import Assistant'


## Enum class for different Encoding Types.
class EncodingType(enum.Enum):
    ## For Symbolic Regression.
    SymReg = 1


## Acts as an Abstract Class / Interface for different Encoding Types.
class EncodingOperations:

    ## Default Constructor
    # @param blueprint Blueprint dictionary to know the rules of encoding.
    def __init__(self, blueprint):
        ## (Dict) Blueprint dictionary to know the rules of encoding.
        self.__blueprint = blueprint

    ## Get private blueprint
    # @return **self.__blueprint**
    def get_blueprint(self):
        return self.__blueprint

    ## (Abstract Method) Generates Encoding for Individual.
    def generate_encoding(self):
        pass

    ## (Abstract Method) Prunes / Cuts the Encoding from a given point / node.
    # @param node Given node / point to prune
    # @param depth Recursive parameter to keep depth
    def prune(self, node):
        pass

    ## (Abstract Method) Replaces the node / point in the Encoding with a replacement node(s).
    # @param node Given node / point to replace.
    # @param replacement Node to replace the given point with.
    # @param cur_node Recursive parameter to keep current node.
    def replace_node(self, node, replacement, node_num):
        pass

    ## (Abstract Method) Random selection of a point / node in the Encoding.
    # @param node Encoding.
    def choose_rand_node(self, node):
        pass

    ## (Abstract Method) Serializes the encoding into a string format.
    # @param node Node / Encoding to be Serialized.
    def serialize_encoding(self, node):
        pass

    ## A static method to create the given Type of Encoding Operations child.
    # @return **EncodingOperations** (Object)
    # @staticmethod
    # def create_encoding_ops(enc_type, blueprint):
    #     if enc_type == EncodingType.SymReg:
    #         return NNEncodingOps(blueprint)
