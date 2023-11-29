import ply.lex as lex

tokens = (
    'OP',
    'ATOM',
    'VAR',
    'NUMBER',
    'LPAREN',
    'RPAREN',
    'LBRACKET',
    'RBRACKET',
    'BAR',
    'COMMA',
    'SEMICOLON',
    'CLAUSE_SEP',
    'PROB_SEP',
    'NEG',
    'DOT',
    'PLUS_MIN',
    'TIMES_DIV'
)

reserved = {
    'is': 'OP',
    'rem': 'TIMES_DIV',
    'div': 'TIMES_DIV'
}

# t_ATOM = r'[a-z]\w*'
t_VAR = r'[A-Z]\w*'
t_NUMBER = r'\d*\.?\d+'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_BAR = r'\|'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = r','
t_SEMICOLON = r';'
t_CLAUSE_SEP = r':-'
t_PROB_SEP = r'::'
t_DOT = r'\.'
t_NEG = r'\\\+'
t_PLUS_MIN = r'[+-]'
t_TIMES_DIV = r'[*/]|rem|div'
t_OP = r'\\==|@<'

t_ignore = ' \t'
t_ignore_COMMENT = r'%.*'

precedence = (
    ('left', 'PLUS_MIN'),
    ('left', 'TIMES_DIV')
)


def t_atom(t):
    r"""[a-z]\w*"""
    t.type = reserved.get(t.value, 'ATOM')
    return t


def t_newline(t):
    r"""\n+"""
    t.lexer.lineno += len(t.value)


def t_error(t):
    raise SyntaxError("Illegal character '%s'" % t.value[0])


lexer = lex.lex()
