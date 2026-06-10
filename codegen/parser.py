""" parse kernel files -> AST."""

import os
from lark import Lark


# Resolve grammar path relative to this file's directory
_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")
grammar = open(_GRAMMAR_PATH).read()
parser = Lark(grammar, start="program")

