from typing import List, Dict
import torch
import pytorch_lightning as pl

import nesy.parser
from nesy.semantics import Semantics
from nesy.term import Clause, Term
from nesy.logic import LogicEngine
from torch import nn
from sklearn.metrics import accuracy_score
from nesy.evaluator import Evaluator

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
                 learning_rate = 0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.program = tuple(program)
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(neural_predicates=neural_predicates, label_semantics=label_semantics)

    def forward(self, tensor_sources: Dict[str, torch.Tensor],  queries: List[Term] | List[List[Term]]):
        # TODO: Note that you need to handle both the cases of single queries (List[Term]), like during training
        #  or of grouped queries (List[List[Term]]), like during testing.
        #  Check how the dataset provides such queries.

        # CASE 1: training
        if isinstance(queries[0], Term):
            # >> STEP C1.1: build and_or_tree -> this will return a list of trees with an and_or_tree for every query
            and_or_trees = self.logic_engine.reason(self.program, queries)
        
            # >> STEP C1.2: evaluate every tree given the image-sequences in tensor_sources. Every query (and so every tree) has a specific image_sequence
            results = self.evaluator.evaluate(tensor_sources, and_or_trees, queries)

        #CASE 2: validation: the queries (the number of queries are equal to the n_digits+1) will be the same except for the different image-sequences (nb equal to the batch size)
        else:
            #With validation, the queries will be the same -> build only 1 tree (based on the query of queries[0])
            and_or_tree = self.logic_engine.reason(self.program, queries[0])
            #Now evaluate this tree for every image_sequences in the batch
            results = []
            for batch_size in range(len(queries)):
                results.append(self.evaluator.evaluate(tensor_sources, and_or_tree, queries[0],batch_size))
        return results

    def training_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I
        
        y_preds = self.forward(tensor_sources, queries) #[tensor(0.1698, grad_fn=<MulBackward0>), tensor(0.3411, grad_fn=<MulBackward0>)] -> tensor([0.4951, 0.3704], grad_fn=<StackBackward0>)
        y_preds_correct_size = torch.stack(y_preds) 
 
        loss = self.bce(y_preds_correct_size, y_true.squeeze())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I
       
        y_preds_proba = self.forward(tensor_sources, queries)
        for inner in y_preds_proba:
            print(torch.stack(inner))
            print(torch.argmax(torch.stack(inner)))
        # calculate the predicted classes by taking the argmax of the predictions
        y_preds = [torch.argmax(torch.stack(inner)) for inner in y_preds_proba]

        accuracy = accuracy_score(y_true, y_preds)
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
