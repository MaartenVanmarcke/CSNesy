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
import torchmetrics

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
                 learning_rate = 0.01, n_classes=2, nb_solutions=3, additional_logs_per_class=False,
                  use_nn_caching=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_predicates = neural_predicates
        self.logic_engine = logic_engine
        self.label_semantics = label_semantics
        self.program = tuple(program)
        self.learning_rate = learning_rate
        self.bce = torch.nn.BCELoss()
        self.evaluator = Evaluator(neural_predicates=neural_predicates, label_semantics=label_semantics,use_nn_caching=use_nn_caching)

        self.n_classes = n_classes
        self.nb_solutions = nb_solutions
        self.additional_logs_per_class = additional_logs_per_class
        if additional_logs_per_class:
            self.acc_indiv = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes,average=None)
            self.acc_sum = torchmetrics.classification.Accuracy(task="multiclass", num_classes=nb_solutions,average=None)

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
            # STEP B.2: evaluate all queries with the images and the constructed trees  + all the predictions per image
            #a = self.evaluator.evaluate(tensor_sources, and_or_tree, queries)
            #b = torch.sum(a, dim = 1)
            #print(b, b.shape)
            return self.evaluator.evaluate(tensor_sources, and_or_tree, queries),self.evaluator.evaluate_for_images(tensor_sources)
            
        
    def training_step(self, I, batch_idx):
        tensor_sources, queries, y_true = I    
        # STEP 1: calculate the outputs of the model given the queries and the tensor_sources. The result is a list of lists. 
        y_preds = self.forward(tensor_sources, queries) 
        # STEP 2: put the y_preds and y_true in the correct format
        correct_size_y_preds =  torch.vstack(y_preds)
        correct_size_y_preds_clamped = torch.clamp(correct_size_y_preds, min=0, max=1)
        correct_size_y_true =   y_true
        # STEP 3: calculate the binary cross entropy loss, this is the mean of the losses of the batch 
        try:
            loss = self.bce(correct_size_y_preds_clamped,correct_size_y_true)    
        except Exception:
            print("correct_size_y_preds", correct_size_y_preds) 
            print("correct_size_y_true", correct_size_y_true)
            loss =self.bce(correct_size_y_preds,correct_size_y_true)    
        self.log("train_loss", loss, on_epoch=True, prog_bar=True) 
        return loss


    def validation_step(self, I, batch_idx):
        tensor_sources, queries, [y_true, ys_trues] = I   
        #STEP 1: calculate the outcome of the model
        y_preds,pred_images = self.forward(tensor_sources, queries)
        #STEP 2: Calculate the accuracy after extracting the predicted value.
        accuracy_sum = accuracy_score(torch.argmax(y_preds, dim = 1), y_true)  

        accuracy_indiv_images = 0
        pred_digits = []
        for i_image in range(len(pred_images)):
            pred_digit = torch.argmax(pred_images[i_image], dim = 1)
            pred_digits.append(pred_digit)
            accuracy_indiv_images += accuracy_score(pred_digit, ys_trues[:,i_image])
            
        accuracy_indiv_images = accuracy_indiv_images/len(pred_images)

        if self.additional_logs_per_class:
            pred_images_tensor = torch.stack(pred_digits).T
            # y_preds_tensor = torch.stack(y_preds)
            indiv_acc_per_class =  self.acc_indiv(pred_images_tensor, ys_trues)
            sum_acc_per_solution = self.acc_sum(y_preds, y_true)
            for i in range(self.n_classes):
                class_name = f'image_acc_of_class_{i}'
                self.log(class_name,indiv_acc_per_class[i], on_epoch=True)
            for i in range(self.nb_solutions):
                if i < self.nb_solutions-1:
                    nb_solution = f'acc_of_solution_{i}'
                else:
                    nb_solution = f'acc_of_solution_illegal'

                self.log(nb_solution, sum_acc_per_solution[i], on_epoch=True)

    
        self.log("val_acc_task", accuracy_sum, on_epoch=True)
        self.log("val_acc_indiv_images", accuracy_indiv_images, on_epoch=True)

        return accuracy_sum

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
