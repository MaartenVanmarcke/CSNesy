from typing import List, Dict
import torch
import pytorch_lightning as pl
import numpy as np
import nesy.parser
from nesy.semantics import Semantics
from nesy.term import Clause, Term
from nesy.logic_optimized import LogicEngine
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
       
        #CASE A: training step if the queries are of type  List[Term]
        if isinstance(queries[0], Term):
            # STEP A.1: construct all the and-or-trees for all given queries
            and_or_tree = self.logic_engine.reason(self.program, queries)
            # STEP A.2: evaluate all queries with the images and the constructed trees 
            return self.evaluator.evaluate(tensor_sources, and_or_tree, queries)

        #CASE B: validation step 
        else:
            # STEP B.1: construct all the and-or-trees for every given query. 
            # >>> Since the queries in a validation step are of type List[List[Term]] we first flatten them
            and_or_tree = self.logic_engine.reason(self.program, list( chain.from_iterable(queries))) 
            # STEP B.2: evaluate all queries with the images and the constructed trees 
            return self.evaluator.evaluate(tensor_sources, and_or_tree, queries)
        
    def training_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I    
        # STEP 1: calculate the outputs of the model given the queries and the tensor_sources. The result is a list of lists. 
        y_preds = self.forward(tensor_sources, queries) 
        # STEP 2: put the y_preds and y_true in the correct format
        correct_size_y_preds =  torch.vstack(y_preds)
        correct_size_y_true =   y_true
        # STEP 3: calculate the binary cross entropy loss, this is the mean of the losses of the batch 
        loss = self.bce(correct_size_y_preds,correct_size_y_true)     
        self.log("train_loss", loss, on_epoch=True, prog_bar=True) 
        return loss


    def validation_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I    
        #STEP 1: calculate the outcome of the model
        y_preds = self.forward(tensor_sources, queries)
        #STEP 2: Calculate the accuracy after extracting the predicted value. 
        accuracy = accuracy_score(torch.argmax(y_preds, dim = 1), y_true)  
        self.log("test_acc", accuracy, on_step=True, on_epoch=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
