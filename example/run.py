import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))

from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic_optimized import ForwardChaining
from nesy.semantics import SumProductSemiring, GodelTNorm, ProductTNorm, LukasieviczTNorm

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


logger = TensorBoardLogger(save_dir="logs/", name="model")  

# STEP 1: define the desired n_digits and n_classes
n_digits = 2
n_classes = 4

#STEP 2: construct the train, test-set and the neural_predicates
task_train = AdditionTask(n=n_digits,n_classes=n_classes)
task_test = AdditionTask(n=n_digits,n_classes=n_classes, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

# STEP 3: define which caching you will use
tree_caching = True
use_nn_caching = True

#STEP 4: define if a validation set during training will be used to see the accuracy evolve over training
# >> BUT if used, adds a lot of operations so makes the program a lot slower
use_validation_set = True   

#STEP 5: define the model + trainer +batch size
model = NeSyModel(program=task_train.program,
                  logic_engine=ForwardChaining(tree_caching),
                  neural_predicates=neural_predicates,
                  label_semantics=SumProductSemiring(),use_nn_caching=use_nn_caching)

if use_validation_set:
    trainer = pl.Trainer(max_epochs=1,logger=logger,log_every_n_steps=1,val_check_interval=1)
else:
    trainer = pl.Trainer(max_epochs=1,logger=logger,log_every_n_steps=1)
batch_size = 64

#STEP 6: fit the model with train and validation data
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=batch_size),
            val_dataloaders=task_test.dataloader(batch_size=batch_size))

