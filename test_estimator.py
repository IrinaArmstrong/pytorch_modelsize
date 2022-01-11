# Basic
import unittest
import logging


# Torch utils
import torch
import torch.nn as nn
from torch.autograd import Variable

import warnings
warnings.simplefilter('ignore')

from pytorch_modelsize import SizeEstimator

class TestSizeEstimator(unittest.TestCase):

    def test_nested_model(self):

        class NestedModel(nn.Module):

            def __init__(self, inputSize, numLayers, nodesPerLayer):
                super(NestedModel, self).__init__()
                self.activation = nn.Sigmoid()
                self.hidden = nn.ModuleList()
                self.hidden.append(nn.Linear(inputSize, nodesPerLayer))
                for i in range(numLayers - 1):
                    self.hidden.append(nn.Linear(nodesPerLayer, nodesPerLayer))
                self.finalFC = nn.Linear(nodesPerLayer, 1)

            def forward(self, x):
                for layer in self.hidden:
                    x = self.activation(layer(x))
                x = self.finalFC(x)
                return x

        nested_model = NestedModel(inputSize=200, numLayers=8, nodesPerLayer=128)
        sample_input = torch.FloatTensor(64, 100, 200)
        print(nested_model)

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
