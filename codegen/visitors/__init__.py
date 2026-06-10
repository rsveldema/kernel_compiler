"""Visitor pattern and AST re-exports for the codegen package."""

from .. import ast
from .visitor import Visitor
from .pretty_printer import PrettyPrinter
from .vulkan_kernel_visitor import VulkanKernelVisitor
from .vulkan_cpp_stub_visitor import VulkanCppStubVisitor
from .resolve_array_indices import ResolveArrayIndicesVisitor, resolve_array_indices

__all__ = [
    # AST nodes (re-exported)
    'Type', 'Int', 'Float', 'FixedSizeVector', 'FlexibleRowsMatrix', 'FixedSizeMatrix',
    'Expression', 'Number', 'Identifier', 'FieldAccess', 'ArrayAccess',
    'LimitExpr', 'BinaryExpr', 'CastExpr', 'NegationExpr',
    'Condition', 'Statement', 'For', 'If', 'Declaration', 'Assignment', 'OverflowCheck',
    'Program', 'WorkgroupProperties', 'SharedDecl',
    # Visitor pattern
    'Visitor', 'PrettyPrinter',
    'VulkanKernelVisitor',
    'VulkanCppStubVisitor',
    'ResolveArrayIndicesVisitor',
    'resolve_array_indices',
]

# Re-export AST nodes at the top-level visitors package
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
