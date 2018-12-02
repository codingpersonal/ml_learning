#
# copyright sigma22 2018
#

import numpy as np
from OutputNode import OutputNode
from InputNode import InputNode
from ReLUNode import ReLUNode

# This generic neutral network stores the nodes in a linked list manner
# each node has the pointer to its previous layer nodes as well as the next layer nodes
# input and output nodes are special nodes where they either have only foeward nodes OR only previous nodes
# this is a fully connected neutral network where every layer is fully connected to the next layer
class NeuralNetworkController:

    # nn_hidden_arch is the array of integers
    def __init__(self, input_layer_count, nn_hidden_arch, output_layer_count):
        # nn_arch is the list of nodes in the neural network.
        # It supports any generic architecture. The first number is the count of input features
        # the final layer is the output layer

        self.input_nodes = []
        self.hidden_nodes = []  # they are stored in the topological order
        self.input_node_count = 3
        self.output_nodes = []

        # ********************************* initialize the input nodes first ***********************
        for i in range(input_layer_count):
            new_node = InputNode(1, i)
            self.input_nodes.append(new_node)

        # ********************************* construct hidden layers ***************************************
        previous_layer = self.input_nodes
        for index in range(len(nn_hidden_arch)):
            curr_layer_nodes = []
            curr_layer_count = nn_hidden_arch[index]
            next_layer_count = nn_hidden_arch[index + 1] if index < (len(nn_hidden_arch) - 1) else output_layer_count
            for j in range(curr_layer_count):
                new_node = ReLUNode(len(previous_layer), next_layer_count, j)
                curr_layer_nodes.append(new_node)
                self.hidden_nodes.append(new_node)

            # connect previous and next node fully now
            for i1 in previous_layer:
                for i2 in curr_layer_nodes:
                    i1.setForwardNode(i2, i2.getNodeNumber())
                    i2.setBackwardNode(i1, i1.getNodeNumber())

            previous_layer = curr_layer_nodes[:]


        # ********************************* construct output layer now ***************************************
        for itr in range(output_layer_count):
            new_node = OutputNode(len(previous_layer), itr, True)
            self.output_nodes.append(new_node)

        # connect the previous layer and output node now
        for i1 in previous_layer:
            for i2 in self.output_nodes:
                i1.setForwardNode(i2, i2.getNodeNumber())
                i2.setBackwardNode(i1, i1.getNodeNumber())


    # this returns the training loss
    # output label is the array of output labels
    def feedTrainingExample(self, input_features, output_label):

        for i in range(len(output_label)):
            self.output_nodes[i].setActualOutputLabel(output_label[i])

        # feed through input layer first
        for i in range(self.input_node_count):
            self.input_nodes[i].accumulateFeedForward(0, input_features[i])
            self.input_nodes[i].feedForward()

        # feed through hidden layers now
        for itr in self.hidden_nodes:
            itr.feedForward()

        # now process the output node
        for itr in self.output_nodes:
            itr.feedForward()

        avg_loss = 0
        # run the backpropagation pass in backward manner
        for itr in reversed(self.output_nodes):
            itr.backpropagate()
            avg_loss += itr.getPredictionLoss()

        for itr in reversed(self.hidden_nodes):
            itr.backpropagate()

        for itr in reversed(self.input_nodes):
            itr.backpropagate()

        return avg_loss/len(output_label)


    def updateNNWeights(self):

        # now process the output node
        for itr in self.output_nodes:
            itr.updateNodeWeights()

        # update  through hidden layers now
        for itr in self.hidden_nodes:
            itr.updateNodeWeights()

        for itr in self.input_nodes:
            itr.updateNodeWeights()


    def printDebugInformation(self):
        for itr in self.output_nodes:
            itr.printInputWeights()


    def trainDummyNN(self):
        # simulate the equation Y = 10 + 3x1 + 5x2 - 2x3
        is_converged = False
        mini_batch_size = 10
        max_batch_count_train = 1000

        while (is_converged == False and max_batch_count_train > 0):
            max_batch_count_train -= 1
            input_feature = np.random.normal(0, 1, (mini_batch_size, 3))

            avg_loss = 0
            for i in range(mini_batch_size):
                output_label = 10 + 30 * input_feature[i][0] + 5 * input_feature[i][1] - 20 * input_feature[i][2]
                avg_loss += self.feedTrainingExample(input_feature[i], [output_label])

            self.updateNNWeights()
            avg_loss /= mini_batch_size

            if max_batch_count_train % 50 == 0:
                # print("avg loss in epoch : ", max_batch_count_train, " is :", avg_loss)
                self.printDebugInformation()
            if abs(avg_loss) < 0.000001:
                print ("data is converged with avg_loss: ", avg_loss)
                is_converged = True

print ("hello world")
nn = NeuralNetworkController(3, [3], 1)
nn.trainDummyNN()

