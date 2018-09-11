#
# copyright sigma22 2018
#

import abc
from BaseNode import BaseNode

class InputNode(BaseNode):

    def __init__(self, output_nodes_count, current_node_number):
        super(InputNode, self).__init__(1, output_nodes_count, current_node_number)

    @abc.abstractmethod
    def feedForward(self):
        self.propagateForward(self.input_node_values[0])

    @abc.abstractmethod
    def backpropagate(self):
        # there is no backpropagation for input node
        return
