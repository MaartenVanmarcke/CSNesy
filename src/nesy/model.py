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
        # print(queries)


        #STEP 1: build a list of the and or tree of every query
        # >> STEP 1A: in the case of training, the queries are List[Term]
        if isinstance(queries[0], Term):
            and_or_tree = self.logic_engine.reason(self.program, queries)
            results, getIndexOfQuery = self.evaluator.evaluate(tensor_sources, and_or_tree, queries)
        
        # >> STEP 2A: evaluate every tree given the images in tensor_sources
            res = []
            for i in range(len(queries)):
                res.append(results[i][getIndexOfQuery(queries[i])])
            return res

        # >> STEP 1B: in the case of testing, the queries are List[List[Term]]
        # else:
        #     results = []
            
        #     for group_queries in queries:
        #          and_or_tree = self.logic_engine.reason(self.program, group_queries)
        #  # >> STEP 2B: evaluate  every tree given the images in tensor_sources
        #          results.append(self.evaluator.evaluate(tensor_sources, and_or_tree, group_queries))
        else:
            and_or_tree = self.logic_engine.reason(self.program, list( chain.from_iterable(queries)))
            results, getIndexOfQuery = self.evaluator.evaluate(tensor_sources, and_or_tree, queries)
            res = torch.zeros_like(results)
            for i in range(results.size()[1]):
                res[:,i] = results[:, getIndexOfQuery(queries[0][i])]
            return res
        
    def training_step(self, I, batch_idx):

        tensor_sources, queries, y_true = I         #y true is always 1 in the case of training -> the training queries are true
        nb_images_per_batch = next(iter(tensor_sources.values())).size()[0]
        # STEP 1: calculate the outputs of the model given the queries and the tensor_sources. The result is a list of lists. 
        # The length of the list is equal to the number of queries and the length of the "inner lists" is equal to the number of tensor_sources
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
        # the true y (of y_trues) gives the index of the correct query (of queries)
        #for example: for the queries = ([addition(tensor(images,0),tensor(images,1),0), addition(tensor(images,0),tensor(images,1),1), addition(tensor(images,0),tensor(images,1),2)], [addition(tensor(images,0),tensor(images,1),0), addition(tensor(images,0),tensor(images,1),1), addition(tensor(images,0),tensor(images,1),2)]) 
        # with y_true = tensor([2, 1])
        #this means that addition(tensor(images,0),tensor(images,1),2) and addition(tensor(images,0),tensor(images,1),1) are correct

        nb_images_per_batch = next(iter(tensor_sources.values())).size()[0]
        #STEP 1: calculate the outcome of the model
        y_preds = self.forward(tensor_sources, queries)
        #STEP 2: reorder the y_preds: select for group of queries the prediction with the highest probability. Do this for every image
        """pred_per_image_per_group = []
        nb_group_queries=len(queries)
        for i in range(nb_group_queries):
            preds_of_group_query = y_preds[i]
            pred_per_image= []
            for index_image in range(nb_images_per_batch):
                preds_of_specific_image = [tensor[index_image] for tensor in preds_of_group_query]
                pred_per_image.append(np.argmax(preds_of_specific_image))
            pred_per_image_per_group.append(pred_per_image)
        correct_size_y_true =  [( y_true[i].repeat(nb_images_per_batch)).tolist() for i in range(len(y_true))]"""
        # print("____")
        # print("y pred:",np.array(pred_per_image_per_group).flatten().tolist())
        # print("y true:",np.array(pred_per_image_per_group).flatten().tolist())
        #accuracy = accuracy_score(np.array(pred_per_image_per_group).flatten().tolist(), np.array(correct_size_y_true).flatten().tolist())  #TODO fix what is given to calculate the y_preds
        accuracy = accuracy_score(torch.argmax(y_preds, dim = 1), y_true)  #TODO fix what is given to calculate the y_preds
        # accuracy = accuracy_score(y_true, y_preds.argmax(dim=-1)) 
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
