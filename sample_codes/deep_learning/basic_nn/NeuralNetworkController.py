#
# copyright sigma22 2018
#

import numpy as np
from OutputNode import OutputNode
from InputNode import InputNode

class NeuralNetworkController:

    def __init__(self, nn_arch):
        # nn_arch is the list of nodes in the neural network.
        # It supports any generic architecture. The first number is the count of input features
        # the final layer is the output layer

        self.input_nodes = {}
        self.input_node_count = 3
        self.output_node = OutputNode(3, 0, True)

        # do a hard coding first
        for i in range(3):
            new_node = InputNode(1, i)
            self.input_nodes[i] = new_node
            new_node.setForwardNode(self.output_node, 0)
            self.output_node.setBackwardNode(new_node, i)

    # this returns the training loss
    def feedTrainingExample(self, input_features, output_label):
        self.output_node.setActualOutputLabel(output_label)
        for i in range(self.input_node_count):
            self.input_nodes[i].accumulateFeedForward(0, input_features[i])
            self.input_nodes[i].feedForward()

        self.output_node.feedForward()

        # run the backward pass
        self.output_node.backpropagate()
        for i in range(self.input_node_count - 1, -1):
            self.input_nodes[i].backpropagate()

        return self.output_node.getPredictionLoss()


    def updateNNWeights(self):
        self.output_node.updateNodeWeights()
        for i in range(self.input_node_count):
            self.input_nodes[i].updateNodeWeights()


    def trainDummyNN(self):
        # simulate the equation Y = 10 + 3x1 + 5x2 - 2x3
        is_converged = False
        mini_batch_size = 100
        max_batch_count_train = 1000

        while (is_converged == False and max_batch_count_train > 0):
            max_batch_count_train -= 1
            input_feature = np.random.normal(0, 1, (mini_batch_size, 3))

            avg_loss = 0
            for i in range(mini_batch_size):
                output_label = 10 + 3 * input_feature[i][0] + 5 * input_feature[i][1] - 2 * input_feature[i][2]
                avg_loss += self.feedTrainingExample(input_feature[i], output_label)

            self.updateNNWeights()
            avg_loss /= mini_batch_size

            if max_batch_count_train % 50 == 0:
                # print("avg loss in epoch : ", max_batch_count_train, " is :", avg_loss)
                self.output_node.printInputWeights()
            if avg_loss < 0.000001:
                print ("data is converged with avg_loss: ", avg_loss)
                is_converged = True

print ("hello world")
nn = NeuralNetworkController([2])
nn.trainDummyNN()

