from nesy.term import Term, Clause
from nesy.parser import parse_term, parse_program
from abc import ABC, abstractmethod
from collections import namedtuple

class LogicEngine(ABC):

    @abstractmethod
    def reason(self, program: list[Clause], queries: list[Term]):
        pass


class ForwardChaining(LogicEngine):

    def reason(self, program: tuple[Clause], queries: list[Term]):
        # TODO: Implement this

        # Dummy example:

        query = parse_term("addition(tensor(images,0), tensor(images,1), 0)")


        Or = lambda x:  None
        And = lambda x: None
        Leaf = lambda x: None
        and_or_tree = Or([
            And([
                Leaf(parse_term("digit(tensor(images,0), 0)")),
                Leaf(parse_term("digit(tensor(images,1), 0)")),
            ])
        ])

        return and_or_tree

'''
Node() abstract ; self.evaluate(dataTensor, semantics)
    - InternalNode(list[Node]) abstract
        - OR(list[Node])
        - AND(list[Node])
    - Leaf abstract
        - NeuralLeaf(model, index, query)  # example: NeuralLeaf(digitModel, 1, 0) for parse_term "digit(tensor(images,1), 0)"
        - FactLeaf(weight)
'''