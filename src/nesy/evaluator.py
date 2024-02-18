import torch
from nesy.tree import Term

class Evaluator():

    def __init__(self, label_semantics, neural_predicates,use_nn_caching=False):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics
        self.use_nn_caching =  use_nn_caching

    def evaluate(self, tensor_sources, and_or_tree, queries): 
        eval_result = and_or_tree.evaluate(tensor_sources,self.label_semantics,self.neural_predicates,self.use_nn_caching)

        # Training phase
        if isinstance(queries[0], Term):
            res = []
            for i in range(len(queries)):
                # Only return the evaluation of the asked query, not of every possible query
                # The and_or_tree keeps the order of the queries, so use it to find the index
                res.append(eval_result[i][and_or_tree.findQuery(queries[i])])
            return res

        # Testing phase
        else:
            res = torch.zeros_like(eval_result)
            for i in range(eval_result.size()[1]):
                res[:,i] = eval_result[:, and_or_tree.findQuery(queries[0][i])]
            return res