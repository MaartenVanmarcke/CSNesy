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
     
        #STEP 1: return and or tree
        # >> STEP 1A: in the case of training, the queries are List[Term]

        if isinstance(queries[0], Term):
            and_or_tree = self.logic_engine.reason(self.program, queries)
            
        # >> STEP 2A: evaluate every tree given the images in tensor_sources
            return self.evaluator.evaluate(tensor_sources, and_or_tree, queries)

        # >> STEP 1B: in the case of testing, the queries are List[List[Term]]
        else:

            # and_or_tree = self.logic_engine.reason(self.program, list( chain.from_iterable(queries))) 
            #TODO question: what is the chaining for? Can we not just construct the tree only once?
                # since the test-queries always are always the same list? (of the possible additions?)
            results = []
            batch_size =  len(queries)
            for batch_nb in range(batch_size):
                and_or_tree = self.logic_engine.reason(self.program,queries[0]) 
                results.append(self.evaluator.evaluate(tensor_sources=tensor_sources,and_or_tree=and_or_tree,queries=queries,image_seq_nb=batch_nb))

            return results
        

    def training_step(self, I, batch_idx):    
        tensor_sources, queries, y_true = I
        y_preds = self.forward(tensor_sources, queries)
        y_preds_correct_size = torch.stack(y_preds) 
        
        loss = self.bce(y_preds_correct_size, y_true.squeeze())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss




    def validation_step(self, I, batch_idx):

        tensor_sources, queries, y_true = I

        y_preds_proba = self.forward(tensor_sources, queries)       
        y_preds = [torch.argmax(torch.stack(inner)) for inner in y_preds_proba]

        accuracy = accuracy_score(y_true, y_preds)
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
