import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))
#
from nesy.tree import ANDNode, ORNode, FactNode, NeuralNode
from torch.nn import Module

leaf1 = FactNode(.8)
leaf2 = FactNode(.4, False)
leaf3 = FactNode(.8, False)
leaf4 = NeuralNode(Module(), 0, 1)

tree = ANDNode(ANDNode(leaf1, ORNode(leaf1, leaf2)),ORNode(leaf3, leaf4))
print(tree)