import torch
from nesy.tree import Term

class Evaluator():

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_tree, queries):   #queries actually not needed
        # print("NB OF TREES:", (len(and_or_trees)))
        # print("NB OF IMAGES:",  tensor_sources["images"].size())
        # print("queries")
        # for i in range(len(queries)):
        #     print(queries[i])
        #     print(and_or_trees[i])
        #     print("-------")
        # for tree in and_or_trees:
        #     print(">>>> ", tree.evaluate(tensor_sources,self.label_semantics,self.neural_predicates))
        
        # print(">> trees toghether: ", [tree.evaluate(tensor_sources,self.label_semantics,self.neural_predicates) for tree in and_or_trees] )
        eval_result = and_or_tree.evaluate(tensor_sources,self.label_semantics,self.neural_predicates)
        if isinstance(queries[0], Term):
            res = []
            for i in range(len(queries)):
                # Only return the evaluation of the asked query, not of every possible query
                res.append(eval_result[i][and_or_tree.findQuery(queries[i])])
            return res

        else:
            res = torch.zeros_like(eval_result)
            for i in range(eval_result.size()[1]):
                res[:,i] = eval_result[:, and_or_tree.findQuery(queries[0][i])]
            return res