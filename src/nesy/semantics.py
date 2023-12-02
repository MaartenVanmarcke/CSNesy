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

    def conjunction(self, a, b):
        return a*b

    def disjunction(self, a, b):
        return a+b

    def negation(self, a):
        return (1-a)

class LukasieviczTNorm(Semantics):

    def conjunction(self, a, b):
        return max(0,a+b-1)

    def disjunction(self, a, b):
        return min(1,a+b)

    def negation(self, a):
       return (1-a)

class GodelTNorm(Semantics):
    # TODO: Implement this

    def conjunction(self, a, b):
        return min(a,b)

    def disjunction(self, a, b):
        return max(a,b)

    def negation(self, a):
        return (1-a)

class ProductTNorm(Semantics):

    def conjunction(self, a, b):
        return a*b

    def disjunction(self, a, b):
        return (a+b) - (a*b)

    def negation(self, a):
        return (1-a)