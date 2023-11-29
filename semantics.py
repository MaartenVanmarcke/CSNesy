from abc import ABC, abstractmethod


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
    # TODO: Implement this

    def conjunction(self, a, b):
        pass

    def disjunction(self, a, b):
        pass

    def negation(self, a):
        pass

class LukasieviczTNorm(Semantics):
    # TODO: Implement this

    def conjunction(self, a, b):
        pass

    def disjunction(self, a, b):
        pass

    def negation(self, a):
        pass

class GodelTNorm(Semantics):
    # TODO: Implement this

    def conjunction(self, a, b):
        pass

    def disjunction(self, a, b):
        pass

    def negation(self, a):
        pass

class ProductTNorm(Semantics):
    # TODO: Implement this

    def conjunction(self, a, b):
        pass

    def disjunction(self, a, b):
        pass

    def negation(self, a):
        pass