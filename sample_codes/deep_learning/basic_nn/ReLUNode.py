#
# copyright sigma22 2018
#

import numpy as np
from BaseNode import BaseNode
import abc

class ReLUNode(BaseNode):

    def __init__(self, input_nodes_count, output_nodes_count, current_node_number):
        super(ReLUNode, self).__init__(input_nodes_count, output_nodes_count, current_node_number)

    @abc.abstractmethod
    def feedForward(self):
        # this should be defined in the derived classes
        # this function would calculate the output value after applying the current
        # weights and activation function

        self.num_mini_batch += 1

        # calculate the linear formula first
        ip_val = np.mumultiply(self.input_node_values, self.input_nodes_weights)
        ip_scalar_val = np.sum(ip_val)
        ip_scalar_val += self.bias_weight

        # apply the activation function
        ip_scalar_val = ip_scalar_val if ip_scalar_val > 0 else 0

        # compute the loss for passing back to previous node
        if ip_scalar_val > 0:
            self.transient_back_propagation_loss_per_input = np.copy(self.input_nodes_weights)
        else:
            self.transient_back_propagation_loss_per_input = np.zeros(self.input_nodes_count)

        # compute the loss for updating the weights of this node
        if ip_scalar_val > 0:
            self.transient_grad_loss_per_input_weight = np.copy(
                self.input_node_values)
            self.transient_grad_loss_for_bias_term = 1
        else:
            self.grad_loss_per_input_weight = np.zeros(self.input_nodes_count)
            self.transient_grad_loss_for_bias_term = 0

        self.propagateForward(ip_scalar_val)

        self.clearCurrentForwardAccumulation()
        return

    @abc.abstractmethod
    def backpropagate(self):

        # compute gradient for current node weights
        self.transient_grad_loss_per_input_weight = self.transient_grad_loss_per_input_weight * self.transient_back_propagation_loss_grad
        self.grad_loss_per_input_weight = self.grad_loss_per_input_weight + self.transient_grad_loss_per_input_weight
        self.grad_loss_for_bias_term += self.transient_back_propagation_loss_grad * self.transient_grad_loss_for_bias_term

        # this would create the backpropagation loss to pass to previous node
        self.transient_back_propagation_loss_per_input = self.transient_back_propagation_loss_per_input * self.transient_back_propagation_loss_grad

        self.propagateLossBackward()
        self.clearCurrentBackwardAccumulation()
