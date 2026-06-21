
from pathlib import Path
from copy import deepcopy

from lark import Tree

from codegen.kast.program import Program
from codegen.visitors import visitor
from codegen.kast.expression import Expression
from codegen.kast.statement import (
    Statement,
    ForLoopWithConditionAndIncrement,
    ForLoopRange,
    Condition,
    If,
    Declaration,
    SharedDecl,
    RawStatement,
)
from codegen.kast.expression import (
    Identifier,
    Number,
    BinaryExpr,
    LimitExpr,
    CastExpr,
    FieldAccess,
    ArrayAccess,
    CallExpr,
    Identifier,
    Number,
    BinaryExpr,
    LimitExpr,
)
from codegen.kast.type import Int, Float
from codegen.parser import parse_search_replace_pattern
from codegen.transforms import transform_statement, transform_expression

class Pattern:
    def __init__(self, filename: str):
        self.filename = filename
        self.search: list[Statement] = []
        self.replace: list[Statement] = []
        self.constraints: Expression | None = None
        self.meta: dict[str, Expression] = {}
        self.init()

    def init(self):
        pattern_tree = parse_search_replace_pattern(Path(self.filename).read_text())
        self.search.clear()
        self.replace.clear()

        for child in pattern_tree.children:
            if not isinstance(child, Tree):
                continue

            if child.data == "search_statements":
                self.search.extend(self._transform_statements(child))
            elif child.data == "replace_statements":
                self.replace.extend(self._transform_statements(child))
            elif child.data == "constraints":
                self.constraints = self._transform_constraints(child)
            elif child.data == "meta":
                self.meta.update(self._transform_meta(child))

    def _transform_statements(self, statements_tree: Tree) -> list[Statement]:
        statements = []

        for child in statements_tree.children:
            if isinstance(child, Tree) and child.data == "statement":
                statement = transform_statement(child)
                if statement is not None:
                    statements.append(statement)

        return statements

    def _transform_constraints(self, constraints_tree: Tree) -> Expression | None:
        for child in constraints_tree.children:
            if isinstance(child, Tree) and child.data == "expression":
                return transform_expression(child)

        return None

    def _transform_meta(self, meta_tree: Tree) -> dict[str, Expression]:
        assignments = {}
        for child in meta_tree.children:
            if not isinstance(child, Tree) or child.data != "meta_assign":
                continue

            name = None
            value = None
            for item in child.children:
                if hasattr(item, "type") and item.type == "IDENT":
                    name = item.value
                elif isinstance(item, Tree) and item.data == "expression":
                    value = transform_expression(item)

            if name is not None and value is not None:
                assignments[name] = value

        return assignments


    def matches(self, node):
        # Placeholder for pattern matching logic
        return False

class PatternStore:
    def __init__(self):
        self.patterns = {}
        self.init()

    def init(self):
        transforms_dir = Path(__file__).resolve().parents[2] / "transforms"
        self.patterns.clear()

        for pattern_path in sorted(transforms_dir.glob("*.tkernel")):
            try:
                pattern = Pattern(str(pattern_path))
                self.patterns[pattern_path.stem] = pattern
            except Exception as e:
                # Skip patterns that can't be parsed (e.g., multi_arg.tkernel uses + which
                # isn't supported in the grammar's for-loop increment rule). These patterns
                # are applied programmatically rather than through generic tree matching.
                print(f"Skipping unparseable pattern {pattern_path.name}: {e}")


class TreeRewriter(visitor.Visitor):
    """
    Visitor that rewrites the AST based on search/replace patterns defined in .tkernel files.
    When seeing a statement node, it checks if any pattern matches it. If a match is found, 
    it replaces the statement with the corresponding replacement statements from the pattern.
    
    When a pattern matches, any wildcard identifiers in the search pattern are bound to the corresponding expression in the AST.
    In the replacement, the wildcard identifiers are replaced with the bound expressions.
    """
    def __init__(self, context):
        self.context = context or {}
        self.pattern_store = PatternStore()

    def visit_program(self, node: Program):
        body_stmts = node.body_stmts        
        node.body_stmts = body_stmts
        for pattern in self.pattern_store.patterns.values():
            self._apply_program_meta(node, pattern)
        return node

    def _apply_program_meta(self, node: Program, pattern: Pattern):
        reduction_chunks = pattern.meta.get("reduction_chunks")
        if isinstance(reduction_chunks, Number):
            node.reduction_chunks = int(reduction_chunks.value)
    
