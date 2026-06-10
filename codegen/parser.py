""" parse kernel files -> AST."""

import os
from lark import Lark
from codegen.ast import Program
from codegen.visitors.pretty_printer import PrettyPrinter
from codegen.transforms import transform


# Resolve grammar path relative to this file's directory
_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")
grammar = open(_GRAMMAR_PATH).read()
parser = Lark(grammar, start="program")


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def compile(filename: str) -> Program:
    print(f"--------------- parsing: {filename} -----------------")
    text = read_file(filename)
    ret = parser.parse(text)
    program = transform(ret)
    return program


def prettyprint(program: Program):
    printer = PrettyPrinter()
    s = program.accept(printer)
    print(s)


