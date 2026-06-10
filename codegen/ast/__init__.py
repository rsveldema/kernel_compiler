"""AST node definitions for code generation.

Each class lives in its own submodule; this package re-exports everything
at the top level so existing imports (``from codegen.ast import Foo``) still work.
"""

from .type import Type, Int, Float, FixedSizeVector, FlexibleRowsMatrix, FixedSizeMatrix, FixedSizeLevelsRowsColsMatrix, FlexibleRowsColsLevelsMatrix
from .expression import Expression, Number, Identifier, FieldAccess, ArrayAccess, LimitExpr, BinaryExpr, CastExpr, NegationExpr
from .statement import Condition, Statement, For, If, Declaration, Assignment, OverflowCheck, SharedDecl
from .workgroup import WorkgroupProperties
from .program import Program

__all__ = [
    # Types
    'Type', 'Int', 'Float',
    'FixedSizeVector', 'FlexibleRowsMatrix', 'FixedSizeMatrix',
    'FixedSizeLevelsRowsColsMatrix', 'FlexibleRowsColsLevelsMatrix',
    # Expressions
    'Expression', 'Number', 'Identifier', 'FieldAccess', 'ArrayAccess',
    'LimitExpr', 'BinaryExpr', 'CastExpr', 'NegationExpr',
    # Statements & conditions
    'Condition', 'Statement', 'For', 'If',
    'Declaration', 'Assignment', 'OverflowCheck', 'SharedDecl',
    # Top-level nodes
    'Program', 'WorkgroupProperties',
]
