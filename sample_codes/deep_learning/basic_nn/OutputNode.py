#
# copyright sigma22 2018
#

import abc
import math
from BaseNode import BaseNode
import numpy as np

class OutputNode(BaseNode):

    def __init__(self, input_nodes_count, current_node_number, is_regression):
        super(OutputNode, self).__init__(input_nodes_count, 0, current_node_number)
        self.is_regression_model = is_regression

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # used during training phase
    def setActualOutputLabel(self, output_val):
        self.current_output_label = output_val
        return

    def getActualOutputLabel(self):
        return self.current_output_label

    def getPredictionLoss(self):
        return (self.getActualOutputLabel() - self.getPredictedLabel())

    def getPredictedLabel(self):
        ip_val = np.multiply(self.input_node_values, self.input_nodes_weights)
        ip_scalar_val = np.sum(ip_val)
        ip_scalar_val += self.bias_weight

        if self.is_regression_model:
            return ip_scalar_val
        else:
            self.sigmoid(ip_scalar_val)

    def printInputWeights(self):
        # print("actual op : ", self.getActualOutputLabel(), " , predicted: ", self.getPredictedLabel())
        print("Input weights are: ", self.bias_weight, " , ", self.input_nodes_weights[0], " , ", self.input_nodes_weights[1], " , ", self.input_nodes_weights[2])

    @abc.abstractmethod
    def feedForward(self):
        # this should be defined in the derived classes
        # this function would calculate the output value after applying the current
        # weights and activation function

        self.num_mini_batch += 1

        # calculate the linear formula first
        ip_val = np.multiply(self.input_node_values, self.input_nodes_weights)
        ip_scalar_val = np.sum(ip_val)
        ip_scalar_val += self.bias_weight

        # todo for classification problem

        # compute the loss for passing back to previous node
        self.transient_back_propagation_loss_per_input = np.copy(self.input_nodes_weights)

        # compute the loss for updating the weights of this node
        self.transient_grad_loss_per_input_weight = np.copy(
            self.input_node_values)
        self.transient_grad_loss_for_bias_term = 1

        self.propagateForward(ip_scalar_val)
        self.clearCurrentForwardAccumulation()
        return

    @abc.abstractmethod
    def backpropagate(self):
        # calculate the true loss gradient for the backpropagation
        self.transient_back_propagation_loss_grad = self.getPredictedLabel() - self.getActualOutputLabel()

        # compute gradient for current node weights
        self.transient_grad_loss_per_input_weight = self.transient_grad_loss_per_input_weight * self.transient_back_propagation_loss_grad
        self.grad_loss_per_input_weight = self.grad_loss_per_input_weight + self.transient_grad_loss_per_input_weight
        self.grad_loss_for_bias_term += self.transient_back_propagation_loss_grad * self.transient_grad_loss_for_bias_term

        # this would create the backpropagation loss to pass to previous node
        self.transient_back_propagation_loss_per_input = self.transient_back_propagation_loss_per_input * self.transient_back_propagation_loss_grad

        self.propagateLossBackward()
        self.clearCurrentBackwardAccumulation()
