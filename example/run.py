import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))
#
from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from nesy.logic import ForwardChaining
from nesy.semantics import SumProductSemiring

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="logs/", name="model")  


task_train = AdditionTask(n_classes=3)
task_test = AdditionTask(n_classes=3, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

tree_caching = True
use_nn_caching = True
model = NeSyModel(program=task_train.program,
                  logic_engine=ForwardChaining(caching_used=tree_caching),
                  neural_predicates=neural_predicates,
                  label_semantics=SumProductSemiring(),use_nn_caching=use_nn_caching)

trainer = pl.Trainer(max_epochs=1,logger=logger,log_every_n_steps=1)
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=2),
            val_dataloaders=task_test.dataloader(batch_size=2))
