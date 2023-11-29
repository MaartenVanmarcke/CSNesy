import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))
#
from src.nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask
from src.nesy.logic import ForwardChaining
from src.nesy.semantics import SumProductSemiring

import torch
import pytorch_lightning as pl

task_train = AdditionTask(n_classes=2)
task_test = AdditionTask(n_classes=2, train=False)

neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})

model = NeSyModel(program=task_train.program,
                  logic_engine=ForwardChaining(),
                  neural_predicates=neural_predicates,
                  label_semantics=SumProductSemiring())

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=model,
            train_dataloaders=task_train.dataloader(batch_size=2),
            val_dataloaders=task_test.dataloader(batch_size=2))
