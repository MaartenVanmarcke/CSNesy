from abc import ABC, abstractmethod
import torch

class Semantics(ABC):

    @abstractmethod
    def conjunction(self, a, b):
        pass

    @abstractmethod
    def disjunction(self, a, b):
        pass

    @abstractmethod
    def negation(self, a):
        pass


class SumProductSemiring(Semantics):

    def conjunction(self, a, b):
        return a*b

    def disjunction(self, a, b):
        return a+b

    def negation(self, a):
        return (1-a)

class LukasieviczTNorm(Semantics):

    def conjunction(self, a, b):
        return torch.maximum(torch.zeros_like(a),a+b-1)

    def disjunction(self, a, b):
        return torch.minimum(torch.ones_like(a),a+b)

    def negation(self, a):
       return (1-a)

class GodelTNorm(Semantics):

    def conjunction(self, a, b):
        return torch.minimum(a,b)

    def disjunction(self, a, b):
        return torch.maximum(a,b)

    def negation(self, a):
        return (1-a)

class ProductTNorm(Semantics):

    def conjunction(self, a, b):
        return a*b

    def disjunction(self, a, b):
        return (a+b) - (a*b)

    def negation(self, a):
        return (1-a)