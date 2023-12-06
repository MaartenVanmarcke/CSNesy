from collections import namedtuple
from collections import Counter

class Term(namedtuple('Term', ['functor', 'arguments'])):

    def __repr__(self):
        if len(self.arguments) == 0:
            return str(self.functor)
        return str(self.functor) + "(" + ",".join([str(a) for a in self.arguments]) + ")"
    
    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, Term)
                and self.functor == __value.functor
                and Counter(self.arguments) == Counter(__value.arguments))
    
    def __contains__(self, __key: object) -> bool:
        if isinstance(__key, Variable):
            return __key in self.arguments
        return super().__contains__(__key)


class Variable(namedtuple('Variable', ['name'])):

    def __repr__(self):
        return str(self.name)
    
    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, Variable)
                and self.name == __value.name)


class Clause(namedtuple('Clause', ['head', 'body'])):

    def __repr__(self):
        return str(self.head) + " :- " + ",".join([str(a) for a in self.body])
    
    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, Clause)
                and self.head == __value.head
                and Counter(self.body) == Counter(__value.body))
    
    def __contains__(self, __key: object) -> bool:
        if isinstance(__key, Variable):
            return __key in self.head or any([__key in b for b in self.body])
        if isinstance(__key, Term):
            return __key == self.head or __key in self.body
        return super().__contains__(__key)


class Fact(namedtuple('Fact', ['term', 'weight'])):

    def __repr__(self):
        if self.weight is None:
            return str(self.term)
        else:
            return str(self.weight) + "::" + str(self.term)
        
    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, Fact)
                and self.term == __value.term
                and self.weight == __value.weight)
