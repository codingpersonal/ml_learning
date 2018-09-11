#
# copyright sigma22 2018
#

import abc
import numpy as np
import random as rd

class BaseNode(object):

    def __init__(self, input_nodes_count, output_nodes_count, current_node_number):
        self.current_node_num = current_node_number

        self.input_node_references = {}
        self.input_node_count = input_nodes_count
        self.input_nodes_weights = np.random.normal(0, 1, input_nodes_count)
        self.input_node_values = np.zeros(input_nodes_count)

        self.transient_back_propagation_loss_grad = 0
        self.transient_back_propagation_loss_per_input = np.zeros(input_nodes_count)   # to backpropagate the loss
                                                                                            # to previous nodes

        self.output_node_references = {}
        self.output_node_count = output_nodes_count

        # bias data
        self.bias_weight = rd.uniform(0, 1)

        # for updating the weights
        self.num_mini_batch = 0
        self.transient_grad_loss_per_input_weight = np.zeros(input_nodes_count)    # this is the loss for this
                                                                                        # training example only
        self.transient_grad_loss_for_bias_term = 0

        self.grad_loss_per_input_weight = np.zeros(input_nodes_count)          # overall loss for mini-batch
        self.grad_loss_for_bias_term = 0

        # learning algorithm
        self.learning_rate = 0.05


    def getNodeNumber(self):
        return self.current_node_num

    # called during initialization
    def setForwardNode(self, new_node, forward_node_index):
        self.output_node_references[forward_node_index] = new_node

    # called during initialization
    def setBackwardNode(self, new_node, backward_node_index):
        self.input_node_references[backward_node_index] = new_node

    # called while computing the forward pass
    def accumulateFeedForward(self, input_index, input_value):
        self.input_node_values[input_index] = input_value

    def accumulateBackpropagationLoss(self, loss_gradient):
        self.transient_back_propagation_loss_grad+= loss_gradient

    # to clear the internal state about forward pass
    def clearCurrentForwardAccumulation(self):
        # self.input_node_values = np.zeros(self.input_node_count)
        # self.transient_grad_loss_per_input = np.zeros(self.input_node_count)
        return

    # to clear the internal state about forward pass
    def clearCurrentBackwardAccumulation(self):
        self.transient_back_propagation_loss_grad = 0
        self.transient_grad_loss_per_input = np.zeros(self.input_node_count)

    def propagateForward(self, forward_value):
        # this will pass the value to all forward nodes
        for index, node in self.output_node_references.iteritems():
            node.accumulateFeedForward(self.current_node_num, forward_value)

    def propagateLossBackward(self):
        # this will pass the loss to all backward nodes
        for index, node in self.input_node_references.iteritems():
            node.accumulateBackpropagationLoss(self.transient_back_propagation_loss_per_input[index])


    @abc.abstractmethod
    # called when this node is processed in feed forward pass
    def feedForward(self):
        # this should be defined in the derived classes
        # this function would calculate the output value after applying the current
        # weights and activation function
        return

    @abc.abstractmethod
    def backpropagate(self):
        # this would create the backpropagation loss wrt to every input node weight and then send it to previous nodes
        return

    def updateNodeWeights(self):
        if self.num_mini_batch > 0:
            self.grad_loss_per_input_weight = self.grad_loss_per_input_weight / self.num_mini_batch
            self.grad_loss_for_bias_term /= self.num_mini_batch

        self.input_nodes_weights = self.input_nodes_weights - self.learning_rate * self.grad_loss_per_input_weight
        self.bias_weight = self.bias_weight - self.learning_rate * self.grad_loss_for_bias_term

        # clear the gradients now
        self.grad_loss_per_input_weight = np.zeros(self.input_node_count)
        self.grad_loss_for_bias_term = 0
        self.num_mini_batch = 0

