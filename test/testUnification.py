import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))
#
from nesy.unifier import Unifier
from nesy.substituer import Substituer
from nesy.term import Term, Clause, Fact, Variable
from nesy.parser import parse_term

unifier = Unifier()
v1 = Variable("_X1")
v2 = Variable("_X2")
#print(unifier.unify(v1, v2))
assert (unifier.unify(v1, v2) == [(v1, v2)])

v1 = parse_term("p(b,Y)")
v2 = parse_term("p(X,X)")
#print(unifier.unify(v1, v2))
assert (unifier.unify(v1, v2) == [(Variable("X"), Term("b",[])), (Variable("Y"), Term("b", []))])

v1 = parse_term("p(X,Y)")
v2 = parse_term("p(Y,X)")
#print(unifier.unify(v1, v2))
assert ((unifier.unify(v1, v2)) == [(Variable("X"), Variable("Y"))])

v1 = parse_term("p(X,X)")
v2 = parse_term("p(b,b)")
#print(unifier.unify(v1, v2))
assert (unifier.unify(v1, v2) == [(Variable("X"), Term("b",[])), (Variable("X"), Term("b", []))])

subst = Substituer()
v1 = Variable("_X1")
v2 = Variable("_X2")
#print(subst._isValidSubstition([(v1, v2)]))                             # True
assert subst._isValidSubstition([(v1, v2)])
#print(subst._isValidSubstition([(v1, v2), (v2,v1)]))                    # False
assert not subst._isValidSubstition([(v1, v2), (v2,v1)])
#print(subst._isValidSubstition([(parse_term("X"),parse_term("f(X)"))])) # False
assert not subst._isValidSubstition([(parse_term("X"),parse_term("f(X)"))])
#print(subst._isValidSubstition([(parse_term("X"),parse_term("f(Y)"))])) # True
assert subst._isValidSubstition([(parse_term("X"),parse_term("f(Y)"))])