import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))

from nesy.model import NeSyModel, MNISTEncoder
from dataset import AdditionTask,BaseConverter
from nesy.logic_optimized import ForwardChaining
from nesy.semantics import SumProductSemiring, GodelTNorm, ProductTNorm, LukasieviczTNorm

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


logger = TensorBoardLogger(save_dir="logs/", name="model")  

# STEP 1: define the desired n_digits and n_classes
n_digits = 3
n_classes = 4

#STEP 2: construct the train, test-set and the neural_predicates

# STEP 3: define which caching you will use
tree_caching = True
use_nn_caching = True
"""
optim = 0
optim_lr = torch.inf
optim_bs = torch.inf"""
optim = []
optim_lr = []
optim_bs = []

#STEP 4: define if a validation set during training will be used to see the accuracy evolve over training
# >> BUT if used, adds a lot of operations so makes the program a lot slower
use_validation_set = False
additional_logs_per_class = False
#STEP 5: define the model + trainer +batch size

for lr in [.005]:# [.0005,.001,.005,.01,.05,.1,.5,1]:
    for bs in  [64]:
        task_train = BaseConverter(n=n_digits,n_classes=n_classes)
        task_test = BaseConverter(n=n_digits,n_classes=n_classes, train=False)

        neural_predicates = torch.nn.ModuleDict({"digit": MNISTEncoder(task_train.n_classes)})
        model = NeSyModel(program=task_train.program,
                        logic_engine=ForwardChaining(tree_caching),
                        neural_predicates=neural_predicates,
                        label_semantics=SumProductSemiring(),
                        learning_rate=lr,
                        use_nn_caching=use_nn_caching,
                        n_classes=n_classes,
                        #nb_solutions=task_train.max_value_term+2,
                        additional_logs_per_class=additional_logs_per_class)

        if use_validation_set:
            trainer = pl.Trainer(max_epochs=1,logger=logger,log_every_n_steps=1,val_check_interval=1)
        else:
            trainer = pl.Trainer(max_epochs=1,logger=logger,log_every_n_steps=1)
        print(task_train.program)
        #STEP 6: fit the model with train and validation data
        import time
        a = time.time()
        trainer.fit(model=model,
                    train_dataloaders=task_train.dataloader(batch_size=bs))#,
                    #val_dataloaders=task_test.dataloader(batch_size=bs))
        b = time.time()-a
        dummy = trainer.validate(model = model,
                            dataloaders = task_test.dataloader(batch_size = bs),
                            verbose = True)
        print("DUMMY", dummy)
        accuracy = dummy[0]["test_acc_sum"]

        """ if accuracy>optim:
            optim = accuracy
            optim_lr = lr
            optim_bs = bs
            print("Improved!")"""
        optim.append(accuracy)
        optim_lr.append(lr)
        optim_bs.append(bs)


        print("| FINAL JUDGEMENT:", accuracy, "| LR =", lr, "| BS =", bs,"|")

import numpy as np
optim = np.asarray(optim)
optim_lr = np.asarray(optim_lr)
optim_bs = np.asarray(optim_bs)

print(np.vstack((optim, optim_lr, optim_bs)).T)


print("| Optim :", optim, "| LR =", optim_lr, "| BS =", optim_bs,"|")
print("LOGIC: ", model.sum)
print("LOGIC DUring FIRST: ", model.sum2)
print("TRAIN:", b)
"""
foraddition task: 3,3:
    | Optim : 0.9723546234509056 | LR = 0.01 | BS = 32 |

for baseconverter: 2,3:
    | Optim : 0.9923736892278361 | LR = 0.005 | BS = 64 |
    """