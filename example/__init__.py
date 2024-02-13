from dataset import Dataset, DataLoader
from dataset import AdditionTask
from nesy.logic_optimized import ForwardChaining
from nesy.parser import parse_program

task_train = AdditionTask(n_classes=2)

program =task_train.program
task_train.n_classes
logic_engine=ForwardChaining()
queries = [parse_program("addition(tensor(images, 0), tensor(images,1), {}).".format(z))[0].term
                       for z in range(2 * 2 - 1)]
logic_engine.reason(program, queries)