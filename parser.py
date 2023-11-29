from ..term import Clause, Term, Fact, Variable

import ply.yacc as yacc
# noinspection PyUnresolvedReferences
from .lexer import tokens, precedence


def p_program(p):
    """program : clause DOT program
    | clause DOT"""
    if len(p) == 4:
        p[0] = [p[1]] + p[3]
    else:
        p[0] = [p[1]]


def p_clause(p):
    """clause : term CLAUSE_SEP arguments"""
    p[0] = Clause(p[1], p[3])


def p_prob_fact(p):
    """ clause : term PROB_SEP term"""
    p[0] = Fact(p[3], p[1])


def p_fact(p):
    """clause : term"""
    p[0] = Fact(p[1], None)


def p_term(p):
    """term : ATOM LPAREN arguments RPAREN"""
    p[0] = Term(p[1], p[3])


def p_arguments(p):
    """arguments : term COMMA arguments
    | term"""
    if len(p) == 4:
        p[0] =tuple((p[1],) + p[3])
    else:
        p[0] = (p[1],)


def p_term_atom(p):
    """term : ATOM
    | NUMBER"""
    p[0] = Term(p[1], tuple())


def p_term_var(p):
    """term : VAR"""
    p[0] = Variable(p[1])


# Error rule for syntax errors
def p_error(p):
    raise SyntaxError("Syntax error while parsing Program", ("", p.lineno, 0, p.lexer.lexdata))


def get_parser():
    return yacc.yacc()