import torch

class Evaluator():

    def __init__(self, label_semantics, neural_predicates):
        self.neural_predicates = neural_predicates
        self.label_semantics = label_semantics

    def evaluate(self, tensor_sources, and_or_trees, queries,image_seq_nb=-1):
    

        #CASE 1: training: The used image-sequence depends on the used query (so the index of the tree)
        if image_seq_nb  < 0:
             result = [and_or_trees[tree_i].evaluate(tensor_sources,self.label_semantics,self.neural_predicates,tree_i) for tree_i in range(len(and_or_trees))]

        #CASE 2: validation: The used image-sequence is given by image_seq_nb 
        # (since with validation the queries are all the same but the evaluated image_sequence varies)
        else: 
             result = [tree.evaluate(tensor_sources,self.label_semantics,self.neural_predicates,image_seq_nb) for tree in and_or_trees]

        # for tree_index in range(len(and_or_trees)):
        #         print('>>>>>>>> Query: ', queries[tree_index])
        #         print('>>>>>>>> TREE: ', and_or_trees[tree_index])
        #         print(">>>>>>>> EVALUATION RES: ", result[tree_index])
            

        return result

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