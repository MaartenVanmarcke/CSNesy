import sys
import pathlib
import os
current = pathlib.Path().absolute()
sys.path.insert(0,os.path.join(current, 'src'))

from nesy.parser import parse_program, parse_clause, parse_term
from nesy.logic_optimized import ForwardChaining
from nesy.term import Clause

import time
start = time.time()

x = """b :- a, d.
e :- b, c.
0.8 :: a.
0.9 :: d."""

progr = parse_program(x)
progr.append(Clause(parse_term("c"), []))

assert str(progr) == "[b :- a,d, e :- b,c, 0.8::a, 0.9::d, c :- ]"
#print(progr)

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("e")]))
#print(tree)
assert (str(tree).split() == ''' - ANDNode : e
        |- ANDNode : b
        |       |- FactNode : a
        |       |        - 0.8
        |       |        - True
        |       |- FactNode : d
        |                - 0.9
        |                - True
        |- FactNode : c
                 - 1
                 - True\n'''.split())

x = """addition(X, Y, Z) :- digit(X,N1), digit(Y,N2), add(N1,N2,Z)."""

progr = parse_program(x)
progr.append(Clause(parse_term("add(0,0,0)"), []))
progr.append(Clause(parse_term("add(0,1,1)"), []))
progr.append(Clause(parse_term("add(1,0,1)"), []))
progr.append(Clause(parse_term("digit(0,1)"), []))
progr.append(Clause(parse_term("digit(1,0)"), []))
progr.append(Clause(parse_term("digit(0,0)"), []))
progr.append(Clause(parse_term("digit(1,1)"), []))

#print(progr)
assert str(progr) == "[addition(X,Y,Z) :- digit(X,N1),digit(Y,N2),add(N1,N2,Z), add(0,0,0) :- , add(0,1,1) :- , add(1,0,1) :- , digit(0,1) :- , digit(1,0) :- , digit(0,0) :- , digit(1,1) :- ]"

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("addition(0,1,1)")]))
#print(tree)
assert (str(tree).split() in ["""- ORNode : addition(0,1,1)
        |- ANDNode : addition(0,1,1)
        |       |- FactNode : digit(0,0)
        |       |        - 1
        |       |        - True
        |       |- ANDNode :
        |               |- FactNode : digit(1,1)
        |               |        - 1
        |               |        - True
        |               |- FactNode : add(0,1,1)
        |                        - 1
        |                        - True
        |- ANDNode : addition(0,1,1)
                |- FactNode : digit(0,1)
                |        - 1
                |        - True
                |- ANDNode :
                        |- FactNode : digit(1,0)
                        |        - 1
                        |        - True
                        |- FactNode : add(1,0,1)
                                 - 1
                                 - True""".split(), """- ORNode : addition(0,1,1)
        |- ANDNode : addition(0,1,1)
        |       |- FactNode : digit(0,1)
        |       |        - 1
        |       |        - True
        |       |- ANDNode :
        |               |- FactNode : digit(1,0)
        |               |        - 1
        |               |        - True
        |               |- FactNode : add(1,0,1)
        |                        - 1
        |                        - True
        |- ANDNode : addition(0,1,1)
                |- FactNode : digit(0,0)
                |        - 1
                |        - True
                |- ANDNode :
                        |- FactNode : digit(1,1)
                        |        - 1
                        |        - True
                        |- FactNode : add(0,1,1)
                                 - 1
                                 - True""".split()
                                 ])



x = """a :- b, c.
a :- c, d."""

progr = parse_program(x)
progr.append(Clause(parse_term("c"), []))
progr.append(Clause(parse_term("b"), []))
progr.append(Clause(parse_term("d"), []))

#print(progr)
assert (str(progr) == "[a :- b,c, a :- c,d, c :- , b :- , d :- ]")

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("a")]))
#print(tree)
assert (str(tree).split() == """- ORNode : a
        |- ANDNode : a
        |       |- FactNode : c
        |       |        - 1
        |       |        - True
        |       |- FactNode : d
        |                - 1
        |                - True
        |- ANDNode : a
                |- FactNode : b
                |        - 1
                |        - True
                |- FactNode : c
                         - 1
                         - True""".split())









x = """b(X) :- a(X).
c(X) :- b(X)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(b)"), []))

#print(progr)
assert (str(progr) == "[b(X) :- a(X), c(X) :- b(X), a(b) :- ]")

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("c(b)")]))
#print(tree)
assert (str(tree).split() == """- ANDNode : c(b)
        |- FactNode : True
        |        - 1
        |        - True
        |- ANDNode : b(b)
                |- FactNode : True
                |        - 1
                |        - True
                |- FactNode : a(b)
                         - 1
                         - True""".split())


x = """f(b) :- a(b), b(b).
c(b) :- f(b), a(b).
d(b) :- c(b), f(b).
d(b) :- a(b), b(b)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(b)"), []))
progr.append(Clause(parse_term("b(b)"), []))

# print(progr)
assert str(progr) == "[f(b) :- a(b),b(b), c(b) :- f(b),a(b), d(b) :- c(b),f(b), d(b) :- a(b),b(b), a(b) :- , b(b) :- ]"

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("d(b)"), parse_term("f(b)")]))
#print(tree)
assert (str(tree).split() == """- ORNode : d(b)
        |- ANDNode : d(b)
        |       |- ANDNode : c(b)
        |       |       |- ANDNode : f(b)
        |       |       |       |- FactNode : a(b)
        |       |       |       |        - 1
        |       |       |       |        - True
        |       |       |       |- FactNode : b(b)
        |       |       |                - 1
        |       |       |                - True
        |       |       |- FactNode : a(b)
        |       |                - 1
        |       |                - True
        |       |- ANDNode : f(b)
        |               |- FactNode : a(b)
        |               |        - 1
        |               |        - True
        |               |- FactNode : b(b)
        |                        - 1
        |                        - True
        |- ANDNode : d(b)
                |- FactNode : a(b)
                |        - 1
                |        - True
                |- FactNode : b(b)
                         - 1
                         - True
- ANDNode : f(b)
        |- FactNode : a(b)
        |        - 1
        |        - True
        |- FactNode : b(b)
                 - 1
                 - True""".split())

x = """f(X, Y) :- a(X), c(Y).
c(Y) :- a(Y), b(Y)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(a)"), []))
progr.append(Clause(parse_term("a(b)"), []))
progr.append(Clause(parse_term("b(b)"), []))

#print(progr)
assert (str(progr) == "[f(X,Y) :- a(X),c(Y), c(Y) :- a(Y),b(Y), a(a) :- , a(b) :- , b(b) :- ]")

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("f(a,b)")]))
#print(tree)
assert (str(tree).split() == """- ANDNode : f(a,b)
        |- FactNode : a(a)
        |        - 1
        |        - True
        |- ANDNode : c(b)
                |- FactNode : a(b)
                |        - 1
                |        - True
                |- FactNode : b(b)
                         - 1
                         - True""".split())

x = """f(X, Y) :- a(X), c(Y).
f(X,X) :- a(X).
f(X,Y) :- g(X,Z), b(Y).
c(Y) :- a(Y), b(Y)."""


progr = parse_program(x)
progr.append(Clause(parse_term("a(a)"), []))
progr.append(Clause(parse_term("a(b)"), []))
progr.append(Clause(parse_term("b(b)"), []))
progr.append(Clause(parse_term("g(a,hehehe)"), []))

#print(progr)
assert (str(progr) == "[f(X,Y) :- a(X),c(Y), f(X,X) :- a(X), f(X,Y) :- g(X,Z),b(Y), c(Y) :- a(Y),b(Y), a(a) :- , a(b) :- , b(b) :- , g(a,hehehe) :- ]")

fc = ForwardChaining()
tree = (fc.reason(progr, [parse_term("f(a,b)")]))
#print(tree)
assert (str(tree).split() == """- ORNode : f(a,b)
        |- ANDNode : f(a,b)
        |       |- FactNode : a(a)
        |       |        - 1
        |       |        - True
        |       |- ANDNode : c(b)
        |               |- FactNode : a(b)
        |               |        - 1
        |               |        - True
        |               |- FactNode : b(b)
        |                        - 1
        |                        - True
        |- ANDNode : f(a,b)
                |- FactNode : g(a,hehehe)
                |        - 1
                |        - True
                |- FactNode : b(b)
                         - 1
                         - True""".split())


print(time.time()-start)