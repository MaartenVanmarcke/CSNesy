from typing import List, Dict
import torch
import pytorch_lightning as pl
import numpy as np
import nesy.parser
from nesy.semantics import Semantics
from nesy.term import Clause, Term
from nesy.logic import LogicEngine
from torch import nn
from sklearn.metrics import accuracy_score
from nesy.evaluator import Evaluator
from itertools import chain

class MNISTEncoder(nn.Module):
    def __init__(self, n):
        self.n = n
        super(MNISTEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 30),
            nn.ReLU(),
            nn.Linear(30, n),
            nn.Softmax(-1))


    def forward(self, x):
        #We flatten the tensor
        original_shape = x.shape
        n_dims = len(original_shape)
        x = x.view(-1, 784)
        o =  self.net(x)

        #We restore the original shape
        o = o.view(*original_shape[0:n_dims-3], self.n)
        return o

class NeSyModel(pl.LightningModule):


    def __init__(self, program : List[Clause],
                 neural_predicates: torch.nn.ModuleDict,
                 logic_engine: LogicEngine,
                 label_semantics: Semantics,
                 learning_rate = 0.001,
                  use_nn_caching=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.program = tuple(program)
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(neural_predicates=neural_predicates, label_semantics=label_semantics,use_nn_caching=use_nn_caching)

    def forward(self, tensor_sources: Dict[str, torch.Tensor],  queries: List[Term] | List[List[Term]],caching=False):
        print("in forward")
        print(queries)
        print(queries[0].__class__)
        #STEP 1: return and or tree
        # >> STEP 1A: in the case of training, the queries are List[Term]
        if isinstance(queries[0], Term):
            print("build tree")
            and_or_tree = self.logic_engine.reason(self.program, queries)
            print("done tree")
            print(and_or_tree)
        # >> STEP 2A: evaluate every tree given the images in tensor_sources
            return self.evaluator.evaluate(tensor_sources, and_or_tree, queries)

        # >> STEP 1B: in the case of testing, the queries are List[List[Term]]
       
        else:
            print("IN ELSE")
            print(self.program)
            print("_______")
            print(list( chain.from_iterable(queries)))
            and_or_tree = self.logic_engine.reason(self.program, list( chain.from_iterable(queries))) 
            for tree in and_or_tree.queries:
                print(tree)
            print()
                #TODO question: what is the chaining for? Can we not just construct the tree only once?
                # since the test-queries always are always the same list? (of the possible additions?)
            return self.evaluator.evaluate(tensor_sources, and_or_tree, queries)
        
    def training_step(self, I, batch_idx):
        print("start training")
        tensor_sources, queries, y_true = I         #y true is always 1 in the case of training -> the training queries are true
        # STEP 1: calculate the outputs of the model given the queries and the tensor_sources. The result is a list of lists. 
        # The length of the list is equal to the number of queries and the length of the "inner lists" is equal to the number of tensor_sources
        y_preds = self.forward(tensor_sources, queries) 

        # STEP 2: put the y_preds and y_true in the correct format
        correct_size_y_preds =  torch.vstack(y_preds)
        correct_size_y_true =   y_true
        # STEP 3: calculate the binary cross entropy loss, this is the mean of the losses of the batch 
        loss = self.bce(correct_size_y_preds,correct_size_y_true)     
        self.log("train_loss", loss, on_epoch=True, prog_bar=True) 
        # assert False
        return loss


    def validation_step(self, I, batch_idx):
        print("start validation")
        tensor_sources, queries, y_true = I    
        # the true y (of y_trues) gives the index of the correct query (of queries)
        #for example: for the queries = ([addition(tensor(images,0),tensor(images,1),0), addition(tensor(images,0),tensor(images,1),1), addition(tensor(images,0),tensor(images,1),2)], [addition(tensor(images,0),tensor(images,1),0), addition(tensor(images,0),tensor(images,1),1), addition(tensor(images,0),tensor(images,1),2)]) 
        # with y_true = tensor([2, 1])
        #this means that addition(tensor(images,0),tensor(images,1),2) and addition(tensor(images,0),tensor(images,1),1) are correct

        nb_images_per_batch = next(iter(tensor_sources.values())).size()[0]
        #STEP 1: calculate the outcome of the model
        y_preds = self.forward(tensor_sources, queries)
        #STEP 2: reorder the y_preds: select for group of queries the prediction with the highest probability. Do this for every image
        accuracy = accuracy_score(torch.argmax(y_preds, dim = 1), y_true)  #TODO fix what is given to calculate the y_preds
        # accuracy = accuracy_score(y_true, y_preds.argmax(dim=-1)) 
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
