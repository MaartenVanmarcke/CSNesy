import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))

from nesy.parser import parse_program, parse_clause, parse_term
from nesy.logic import ForwardChaining
from nesy.term import Clause

x = """b :- a, d.
e :- b, c.
0.8 :: a.
0.9 :: d."""

progr = parse_program(x)
progr.append(Clause(parse_term("c"), []))

print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("e")]))
print(tree)

x = """addition(X, Y, Z) :- digit(X,N1), digit(Y,N2), add(N1,N2,Z)."""

progr = parse_program(x)
progr.append(Clause(parse_term("add(0,0,0)"), []))
progr.append(Clause(parse_term("add(0,1,1)"), []))
progr.append(Clause(parse_term("add(1,0,1)"), []))
progr.append(Clause(parse_term("digit(0,1)"), []))
progr.append(Clause(parse_term("digit(1,0)"), []))
progr.append(Clause(parse_term("digit(0,0)"), []))
progr.append(Clause(parse_term("digit(1,1)"), []))

print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("addition(0,1,1)")]))
print(tree)



x = """a :- b, c.
a :- c, d."""

progr = parse_program(x)
progr.append(Clause(parse_term("c"), []))
progr.append(Clause(parse_term("b"), []))
progr.append(Clause(parse_term("d"), []))

print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("a")]))
print(tree)










x = """b(X) :- a(X).
c(X) :- b(X)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(b)"), []))

print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("c(b)")]))
print(tree)



x = """f(b) :- a(b), b(b).
c(b) :- f(b), a(b).
d(b) :- c(b), f(b).
d(b) :- a(b), b(b)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(b)"), []))
progr.append(Clause(parse_term("b(b)"), []))

print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("d(b)"), parse_term("f(b)")]))
print(tree)


x = """f(X, Y) :- a(X), c(Y).
c(Y) :- a(Y), b(Y)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(a)"), []))
progr.append(Clause(parse_term("a(b)"), []))
progr.append(Clause(parse_term("b(b)"), []))

print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("f(a,b)")]))
print(tree)

x = """f(X, Y) :- a(X), c(Y).
f(X,X) :- a(X).
f(X,Y) :- g(X,Z), b(Y).
c(Y) :- a(Y), b(Y)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(a)"), []))
progr.append(Clause(parse_term("a(b)"), []))
progr.append(Clause(parse_term("b(b)"), []))
progr.append(Clause(parse_term("g(a,hehehe)"), []))

print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("f(a,b)")]))
print(tree)



