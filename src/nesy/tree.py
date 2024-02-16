from abc import ABC, abstractmethod

import torch
from nesy.semantics import Semantics
from torch import Tensor
from torch.nn import Module
from nesy.term import Term

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
    def __init__(self, name="") -> None:
        self.name = name
        # initialze cache that stores the result of the nn during one training 
        # This is useful in a tree where a has multiple nn-nodes with the same neural network
        self.nn_results_cache = dict()
        super().__init__()

    @abstractmethod
    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,neural_predicates:torch.nn.ModuleDict,image_seq_nb:int,results_nn_cache:dict|None):
        """
        An abstract method that evaluates this node for an input (tensor_sources), for some given semantics and some neural predicates.
        """
        pass

class InternalNode(AndOrTreeNode):
    ''' 
    An abstract class representing a generic internal node of an And Or Tree.\n
    Each internal node has 2 child nodes.
    '''
    
    def __init__(self, child1: AndOrTreeNode, child2: AndOrTreeNode, name:str="") -> None:
        super().__init__(name)
        self.child1 = child1
        self.child2 = child2

    @abstractmethod
    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,image_seq_nb:int,neural_predicates:torch.nn.ModuleDict):
        pass

    def __repr__(self) -> str:
        res = "- " + str(self.__class__.__name__) + " : " + self.name
        c1 = str(self.child1).split('\n')
        c2 = str(self.child2).split('\n')
        res += "\n\t|" + "\n\t|".join(c1) + "\n\t|" + "\n\t".join(c2) 
        return res

class ORNode(InternalNode):
    ''' 
    A class representing an OR node of an And Or Tree.\n
    '''

    def __init__(self, child1: AndOrTreeNode, child2: AndOrTreeNode, name:str="") -> None:
        super().__init__(child1, child2, name)

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,neural_predicates:torch.nn.ModuleDict,image_seq_nb:int,results_nn_cache:dict|None):
        """
        The evaluation of an OR node consists of the evaluation of both child nodes and a disjunction of their results as specified by the given semantics.
        """
        res1 = self.child1.evaluate(tensor_sources, semantics,neural_predicates,image_seq_nb,results_nn_cache)
        res2 = self.child2.evaluate(tensor_sources, semantics,neural_predicates,image_seq_nb,results_nn_cache)
        # print("________")
        # print(self)
        # print(res1, "+", res2)
        # print("result: ", semantics.disjunction(res1, res2))
        return semantics.disjunction(res1, res2)
    

class ANDNode(InternalNode):
    ''' 
    A class representing an AND node of an And Or Tree.\n
    '''

    def __init__(self, child1: AndOrTreeNode, child2: AndOrTreeNode, name:str="") -> None:
        super().__init__(child1, child2, name)

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,neural_predicates:torch.nn.ModuleDict,image_seq_nb:int,nn_results_cache:dict|None):
        """
        The evaluation of an AND node consists of the evaluation of both child nodes and a condjunction of their results as specified by the given semantics.
        """
        res1 = self.child1.evaluate(tensor_sources, semantics,neural_predicates,image_seq_nb,nn_results_cache)
        res2 = self.child2.evaluate(tensor_sources, semantics,neural_predicates,image_seq_nb,nn_results_cache)
        # print("________")
        # print(self)
        # print(res1, "*", res2)
        # print("result: ", semantics.conjunction(res1, res2))
        return semantics.conjunction(res1, res2)


class LeafNode(AndOrTreeNode):
    ''' 
    An abstract class representing a generic leaf node of an And Or Tree.\n
    A leaf node can be a weighted fact or a neural fact.
    '''

    @abstractmethod
    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,neural_predicates,image_seq_nb:int,nn_results_cache:dict|None):
        pass

class FactNode(LeafNode):
    ''' 
    A class representing a weighted fact leaf node of an And Or Tree.\n
    Such a leaf node has a weight and a bool that denotes whether this is a positive or a negated fact.
    '''

    def __init__(self, weight, positive: bool = True, name:str="") -> None:
        super().__init__(name)
        if weight < 0 or weight > 1:
            raise ValueError("The given weight is not between 0 and 1.")
        self.weight = weight
        self.positive = positive
        self.str = str

    def __repr__(self) -> str:
        res ="- " +  str(self.__class__.__name__) + " : " + self.name
        res += "\n\t - " + str(self.weight) + "\n\t - " + str(self.positive) # TODO
        return res

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,neural_predicates,image_seq_nb:int,nn_results_cache:dict|None):
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
    def __init__(self, model: Module, index: int, query, name:str="") -> None: #TODO I think module should actually be string (defined in "logic") = name of NN
        super().__init__(name)
        self.model=model
        if isinstance(model, str):
            self.model = model
        else:
            self.model=model[0]
        if isinstance(index, int):
             self.index = index
        else: self.index =int(index[0])
        if isinstance(query, int):
             self.query = query
        else: self.query =int(query[0])

    def __repr__(self) -> str:
        res = "- " + str(self.__class__.__name__) + " : " + self.name
        res += "\n\t - " + str(self.model) + "\n\t - " + str(self.index) + "\n\t - " + str(self.query)
        return res

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,neural_predicates,image_seq_nb:int,nn_results_cache:dict|None): #TODO I think the neural_predicates should be given as well _>access NN by using name of NN
        """
        The evaluation of a neural fact leaf node is the output of the neural network of this node
        for the given input.
        """
        
        # STEP 1 if nn_results_cache is used, check if the neural network has not already been evaluated
        if nn_results_cache is not None:
            cached = nn_results_cache.get(str(image_seq_nb) + ":" + str(self.index))
            # print("from cache: ", cached)
            if cached is not None:
                return cached[self.query]
            
        # STEP 2: get the neural network of the leaf
        network = neural_predicates[self.model]

        #STEP 3: select the image from the tensor_sources
        image =tensor_sources["images"][image_seq_nb,self.index]

        #STEP 4: get the predictions of the NN: this is the probability that the image belongs to every class 
        pred_of_network = network.forward(image)
        
        #STEP 5: if nn_results_cache is used, store the obtained result of the nn
        if nn_results_cache is not None:
            nn_results_cache[str(image_seq_nb) + ":" + str(self.index)] = pred_of_network
        # print("PRED OF NN", pred_of_network)

        # print("RETURN FROM NN LEAF", pred_of_network[self.query]   )
        #STEP 6: return the relevant prediction 
        return pred_of_network[self.query]   
        ## TODO: how to check if it is between 0 and 1? -> always the case since a softmax is the last layer of the nn


class AndOrTree():
    ''' 
    A class representing an AND OR Tree with multiple roots (one per query).
    The attributes are:
        - self.queries: the list of the root nodes.
        - self.terms: the corresponding terms of those root nodes.
    '''
    def __init__(self, queries: list[AndOrTreeNode], terms: list[Term]) -> None:
        
        if len(queries) != len(terms):
            raise ValueError("Invalid arguments.")
        self.queries = queries
        self.terms = terms

    def evaluate(self, tensor_sources: Tensor, semantics: Semantics,neural_predicates:torch.nn.ModuleDict,queries, image_seq_nb=-1,use_nn_caching=False):
        """
        Evaluates the full and-or-tree for the given inputs.
        """
        res = torch.zeros((tensor_sources["images"].size()[0], len(queries)))

        res = []

        # Evaluate each query
        if image_seq_nb < 0:
            for i in range(len(queries)):
                if use_nn_caching:
                    nn_cache = dict()   #TODO where to initialze this?
                else:
                    nn_cache = None
                # print("<< ", i)
                    
                res.append(self.queries[self.findQuery(queries[i])].evaluate(tensor_sources, semantics, neural_predicates,i,nn_cache))
               
        else:
             for i in range(len(queries[0])):
                if use_nn_caching:
                    nn_cache = dict()  
                else:
                    nn_cache = None
                res.append(self.queries[self.findQuery(queries[0][i])].evaluate(tensor_sources, semantics, neural_predicates,image_seq_nb,nn_cache))
            
        return res
    
    def __repr__(self) -> str:
        res = ""
        for q in self.queries:
            res += q.__repr__()
            res += "\n"
        return res
    
    def findQuery(self, query: Term):
        """
        Returns the index of the given query in the self.terms list.
        """
        return self.terms.index(query)