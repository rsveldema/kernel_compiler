"""parse kernel files -> AST."""

import os
from pathlib import Path
from lark import Lark

from codegen.transforms import transform as _transform


# Resolve grammar path relative to this file's directory
_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")
grammar = open(_GRAMMAR_PATH).read()
parser = Lark(grammar, start="program")
pattern_parser = Lark(grammar, start="search_replace_pattern")


def parse(path_or_text):
    """Parse kernel content and return a Program AST node.
    
    Accepts either a file path (str or Path) or raw kernel text.
    If the argument is a file path, reads the file contents first.
    """
    if isinstance(path_or_text, (str, Path)) and os.path.isfile(path_or_text):
        with open(path_or_text) as f:
            content = f.read()
    else:
        content = path_or_text
    
    tree = parser.parse(content)
    return _transform(tree)


def parse_search_replace_pattern(text):
    """Parse a search/replace pattern file and return its Lark tree."""
    return pattern_parser.parse(text)

__all__ = ['parse', 'parse_search_replace_pattern']
