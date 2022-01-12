# Basic
import sys
import unittest

# Torch utils
import torch
import torch.nn as nn

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout)

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
        logging.info(nested_model)

        estimator = SizeEstimator(nested_model, input_size=sample_input.size())
        total_size = estimator.estimate_total_size()
        logging.info(f"\nTotal: {total_size}")


if __name__ == '__main__':
    unittest.main()
