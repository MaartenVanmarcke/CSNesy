from collections import namedtuple


class Term(namedtuple('Term', ['functor', 'arguments'])):

    def __repr__(self):
        if len(self.arguments) == 0:
            return str(self.functor)
        return str(self.functor) + "(" + ",".join([str(a) for a in self.arguments]) + ")"


class Variable(namedtuple('Variable', ['name'])):

    def __repr__(self):
        return str(self.name)


class Clause(namedtuple('Clause', ['head', 'body'])):

    def __repr__(self):
        return str(self.head) + " :- " + ",".join([str(a) for a in self.body])


class Fact(namedtuple('Fact', ['term', 'weight'])):

    def __repr__(self):
        if self.weight is None:
            return str(self.term)
        else:
            return str(self.weight) + "::" + str(self.term)
