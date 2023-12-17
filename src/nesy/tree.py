from abc import ABC, abstractmethod
from semantics import Semantics
from torch import Tensor
from torch.nn import Module

'''
Node() abstract ; self.evaluate(dataTensor, semantics)
    - InternalNode(list[Node]) abstract
        - OR(list[Node])
        - AND(list[Node])
    - Leaf abstract
        - NeuralLeaf(model, index, query)  # example: NeuralLeaf(digitModel, 1, 0) for parse_term "digit(tensor(images,1), 0)"
        - FactLeaf(weight)
'''

class AndOrTreeNode(ABC):
    ''' 
    An abstract class representing a generic node of an And Or Tree.\n
    The tree is represented by the root node.
    '''

    @abstractmethod
    def evaluate(self, tensor_sources: Tensor, semantics: Semantics):
        """
        An abstract method that evaluates this node for an input (tensor_sources) and for some given semantics.
        """
        pass

class InternalNode(AndOrTreeNode):
    ''' 
    An abstract class representing a generic internal node of an And Or Tree.\n
    Each internal node has 2 child nodes.
    '''
    
    def __init__(self, child1: AndOrTreeNode, child2: AndOrTreeNode) -> None:
        super().__init__()
        self.child1 = child1
        self.child2 = child2

    @abstractmethod
    def evaluate(self, tensor_sources: Tensor, semantics: Semantics):
        pass

class ORNode(InternalNode):
    ''' 
    A class representing an OR node of an And Or Tree.\n
    '''

    def __init__(self, child1: AndOrTreeNode, child2: AndOrTreeNode) -> None:
        super().__init__(child1, child2)

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics):
        """
        The evaluation of an OR node consists of the evaluation of both child nodes and a disjunction of their results as specified by the given semantics.
        """
        res1 = self.child1.evaluate(tensor_sources, semantics)
        res2 = self.child2.evaluate(tensor_sources, semantics)
        return semantics.disjunction(res1, res2)
    

class ANDNode(InternalNode):
    ''' 
    A class representing an AND node of an And Or Tree.\n
    '''

    def __init__(self, child1: AndOrTreeNode, child2: AndOrTreeNode) -> None:
        super().__init__(child1, child2)

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics):
        """
        The evaluation of an AND node consists of the evaluation of both child nodes and a condjunction of their results as specified by the given semantics.
        """
        res1 = self.child1.evaluate(tensor_sources, semantics)
        res2 = self.child2.evaluate(tensor_sources, semantics)
        return semantics.conjunction(res1, res2)


class LeafNode(AndOrTreeNode):
    ''' 
    An abstract class representing a generic leaf node of an And Or Tree.\n
    A leaf node can be a weighted fact or a neural fact.
    '''
    
    @abstractmethod
    def evaluate(self, tensor_sources: Tensor, semantics: Semantics):
        pass

class FactNode(LeafNode):
    ''' 
    A class representing a weighted fact leaf node of an And Or Tree.\n
    Such a leaf node has a weight and a bool that denotes whether this is a positive or a negated fact.
    '''

    def __init__(self, weight, positive: bool = True) -> None:
        super().__init__()
        if weight < 0 or weight > 1:
            raise ValueError("The given weight is not between 0 and 1.")
        self.weight = weight
        self.positive = positive

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics):
        """
        The evaluation of a weighted fact leaf node is: 
            - the weight of this node if this fact is positive; 
            - the negation of the weight of this node if this fact is negative.
        """
        if not self.positive:
            return semantics.negation(self.weight)
        return self.weight

class NeuralNode(LeafNode):
    ''' 
    A class representing a neural fact leaf node of an And Or Tree.\n
    Such a leaf node has a network model, an index that denodes which tensor is the input of this node 
     and a query that denotes for which result of the network we want the probability.
    '''
    def __init__(self, model: Module, index: int, query) -> None:
        super().__init__()
        self.model = model
        self.index = index  
        self.query = query

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics):
        """
        The evaluation of a neural fact leaf node is the output of the neural network of this node
        for the given input.
        """
        return self.model(tensor_sources["images"][:,self.index])[:, self.query]
        ## TODO: how to check if it is between 0 and 1? -> always the case since a softmax is the last layer of the nn

