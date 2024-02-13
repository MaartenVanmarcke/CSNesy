from collections import namedtuple
from collections import Counter

class Term(namedtuple('Term', ['functor', 'arguments'])):

    def __repr__(self):
        if len(self.arguments) == 0:
            return str(self.functor)
        return str(self.functor) + "(" + ",".join([str(a) for a in self.arguments]) + ")"
    
    def __eq__(self, __value: object) -> bool:
        if (isinstance(__value, Term) and self.functor == __value.functor and len(self.arguments) == len(__value.arguments)):
            for i in range(len(self.arguments)):
                if self.arguments[i] != __value.arguments[i]:
                    return False
            return True
        else:
            return False
    
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
        if (isinstance(__value, Clause)
                and self.head == __value.head
                and len(self.body) == len(__value.body)):
            for i in range(len(self.body)):
                if self.body[i] != __value.body[i]:
                    return False
            return True
        else:
            return False
    
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
