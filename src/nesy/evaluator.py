import torch

class Evaluator():

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_trees, queries):   #queries actually not needed
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
        eval_result=[tree.evaluate(tensor_sources,self.label_semantics,self.neural_predicates) for tree in and_or_trees]
        end_result = []
        print(eval_result)
        print(queries)
        for i in range(len(queries)):
            end_result.append(eval_result[i][i])

        return end_result   #TODO also give  self.neural_predicates


        # Our dummy And-Or-Tree (addition(img0, img1,0) is represented by digit(img0,0) AND digit(img1,0)
        # The evaluation is:
        # p(addition(img0, img1,0)) = p(digit(img0,0) AND digit(img1,0)) =
        p_digit_0_0 = self.neural_predicates["digit"](tensor_sources["images"][:,0])[:,0]
        p_digit_1_0 = self.neural_predicates["digit"](tensor_sources["images"][:,1])[:,0]
        p_sum_0 =  p_digit_0_0 * p_digit_1_0

        # Here we trivially return the same value (p_sum_0[0]) for each of the queries to make the code runnable
        if isinstance(queries[0], list):
            res = [torch.stack([p_sum_0[0] for q in query]) for query in queries]
        else:
            res = [p_sum_0[0] for query in queries]
        return torch.stack(res)