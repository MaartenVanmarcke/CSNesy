from ..term import Clause, Term
from .parser import get_parser


def parse_term(text: str) -> Term:
    return get_parser().parse(text + '.')[0].term

def parse_clause(text: str) -> Clause:
    return get_parser().parse(text + '.')[0]


def parse_program(text: str, debug=False) -> list[Clause]:
    return get_parser().parse(text, debug=debug)
