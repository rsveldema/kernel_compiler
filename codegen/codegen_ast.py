"""Re-exports for the codegen AST and visitor modules."""

from . import ast
from .visitors.visitor import Visitor
from .visitors.pretty_printer import PrettyPrinter

__all__ = [
    'ast', 'Visitor', 'PrettyPrinter',
    # AST nodes
    'Type', 'Int', 'Float', 'FixedSizeVector', 'FlexibleRowsMatrix', 'FixedSizeMatrix',
    'Expression', 'Number', 'Identifier', 'FieldAccess', 'ArrayAccess',
    'LimitExpr', 'BinaryExpr', 'CastExpr', 'NegationExpr',
    'Condition', 'Statement', 'For', 'If', 'Declaration', 'Assignment', 'OverflowCheck',
    'Program', 'WorkgroupProperties', 'SharedDecl',
]

# Re-export all AST node classes for backward compatibility
Type = ast.Type
Int = ast.Int
Float = ast.Float
FixedSizeVector = ast.FixedSizeVector
FlexibleRowsMatrix = ast.FlexibleRowsMatrix
FixedSizeMatrix = ast.FixedSizeMatrix
Expression = ast.Expression
Number = ast.Number
Identifier = ast.Identifier
FieldAccess = ast.FieldAccess
ArrayAccess = ast.ArrayAccess
LimitExpr = ast.LimitExpr
BinaryExpr = ast.BinaryExpr
CastExpr = ast.CastExpr
NegationExpr = ast.NegationExpr
Condition = ast.Condition
Statement = ast.Statement
For = ast.For
If = ast.If
Declaration = ast.Declaration
Assignment = ast.Assignment
OverflowCheck = ast.OverflowCheck
Program = ast.Program
WorkgroupProperties = ast.WorkgroupProperties
SharedDecl = ast.SharedDecl
