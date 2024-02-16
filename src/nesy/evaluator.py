import torch
from nesy.tree import Term

class Evaluator():

    def __init__(self, label_semantics, neural_predicates,use_nn_caching=False):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics
        self.use_nn_caching =  use_nn_caching

    def evaluate(self, tensor_sources, and_or_tree, queries,image_seq_nb=-1): 
        return and_or_tree.evaluate(tensor_sources=tensor_sources,semantics=self.label_semantics,neural_predicates=self.neural_predicates,queries=queries,image_seq_nb=image_seq_nb,use_nn_caching=self.use_nn_caching)

        # eval_result =  and_or_tree.evaluate(tensor_sources=tensor_sources,semantics=self.label_semantics,neural_predicates=self.neural_predicates,image_seq_nb=image_seq_nb,use_nn_caching=self.use_nn_caching)
        # eval_result_tensor = torch.tensor(eval_result)
        # print("EVALUATION")
        # print(">>", eval_result)
        # print(">>", eval_result_tensor)
        # if isinstance(queries[0], Term):
        #         res = []
        #         for i in range(len(queries)):
        #             # Only return the evaluation of the asked query, not of every possible query
        #             res.append(eval_result_tensor[i][and_or_tree.findQuery(queries[i])])
        #         return res

        # else:
        #         res = torch.zeros_like(eval_result_tensor)
        #         for i in range(eval_result_tensor.size()[1]):
        #             res[:,i] = eval_result_tensor[:, and_or_tree.findQuery(queries[0][i])]
        #         return res