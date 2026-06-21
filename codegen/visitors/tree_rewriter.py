from pathlib import Path
import ast as py_ast
import copy
import operator
import re

from lark import Tree

from codegen.kast.expression import BinaryExpr, Expression, Identifier, Number, UnaryMinusExpr, WildcardExpression
from codegen.kast.program import Program
from codegen.kast.statement import Statement, WildcardStatement
from codegen.parser import parse_search_replace_pattern
from codegen.transforms import transform_expression, transform_statement
from codegen.visitors import visitor
from codegen.visitors.pattern_match_visitor import PatternMatchVisitor


class Pattern:
    def __init__(self, filename: str):
        self.filename = filename
        self.search: list[Statement] = []
        self.replace: list[Statement] = []
        self.constraints: Expression | None = None
        self.meta: dict[str, Expression] = {}
        self.target_header: str | None = None
        self.wildcard_statement_map: dict[str, Statement] = {}
        self.wildcard_expression_map: dict[str, Expression] = {}
        self.init()

    def init(self):
        text = Path(self.filename).read_text()
        self._load_meta_from_text(text)

        try:
            pattern_tree = parse_search_replace_pattern(text)
        except Exception:
            return

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

    def _load_meta_from_text(self, text: str) -> None:
        match = re.search(r'\boptimizes\s*\(\s*"([^"]+)"\s*\)', text)
        if match:
            self.target_header = match.group(1)

        meta_text = self._extract_meta_block(text)
        if meta_text is None:
            return
        self.meta.update(self._parse_meta_assignments(meta_text))

    def _extract_meta_block(self, text: str) -> str | None:
        start = text.rfind("meta")
        if start == -1:
            return None

        open_brace = text.find("{", start)
        if open_brace == -1:
            return None

        depth = 0
        for index in range(open_brace, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return None

    def _parse_meta_assignments(self, meta_text: str) -> dict[str, Expression]:
        assignments = {}
        body_start = meta_text.find("{")
        body_end = meta_text.rfind("}")
        if body_start == -1 or body_end == -1 or body_end <= body_start:
            return assignments

        body = meta_text[body_start + 1:body_end]
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;,\n}]+)", body):
            name = match.group(1)
            raw_value = match.group(2).strip()
            value = self._parse_meta_value(raw_value)
            if value is not None:
                assignments[name] = value

        return assignments

    def _parse_meta_value(self, raw_value: str) -> Expression | None:
        if re.fullmatch(r"[0-9]+", raw_value):
            return Number(int(raw_value))
        if re.fullmatch(r"([0-9]+\.[0-9]*|\.[0-9]+)(f|F)?", raw_value):
            return Number(float(raw_value.rstrip("fF")))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw_value):
            return Identifier(raw_value)
        folded_value = self._fold_python_constant(raw_value)
        if folded_value is not None:
            return Number(folded_value)
        return None

    def _fold_python_constant(self, raw_value: str):
        try:
            node = py_ast.parse(raw_value.rstrip("fF"), mode="eval")
        except SyntaxError:
            return None

        try:
            return self._eval_python_constant_node(node.body)
        except ValueError:
            return None

    def _eval_python_constant_node(self, node):
        binary_ops = {
            py_ast.Add: operator.add,
            py_ast.Sub: operator.sub,
            py_ast.Mult: operator.mul,
            py_ast.Div: operator.truediv,
            py_ast.FloorDiv: operator.floordiv,
            py_ast.Mod: operator.mod,
        }

        if isinstance(node, py_ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, py_ast.UnaryOp) and isinstance(node.op, py_ast.USub):
            return -self._eval_python_constant_node(node.operand)
        if isinstance(node, py_ast.BinOp):
            op = binary_ops.get(type(node.op))
            if op is None:
                raise ValueError
            return op(
                self._eval_python_constant_node(node.left),
                self._eval_python_constant_node(node.right),
            )
        raise ValueError

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

    def matches(self, node: Program) -> bool:
        matcher = PatternMatchVisitor()
        if not matcher.matches_statements(self.search, node.body_stmts):
            self.wildcard_statement_map.clear()
            self.wildcard_expression_map.clear()
            return False

        self.wildcard_statement_map = matcher.wildcard_statement_map
        self.wildcard_expression_map = matcher.wildcard_expression_map
        return True

    def apply(self, node: Program):
        node.body_stmts = [self._clone_statement(statement) for statement in self.replace]
        print(f"TREE TRANSFORMED: {node.header}")
        node.tree_transformed = True

    def _clone_statement(self, statement: Statement) -> Statement:
        if isinstance(statement, WildcardStatement):
            return copy.deepcopy(self.wildcard_statement_map[statement.name])
        return self._replace_wildcards(copy.deepcopy(statement))

    def _replace_wildcards(self, node):
        if isinstance(node, WildcardExpression):
            return copy.deepcopy(self.wildcard_expression_map[node.name])

        if isinstance(node, list):
            for index, item in enumerate(node):
                if isinstance(item, WildcardStatement):
                    node[index] = copy.deepcopy(self.wildcard_statement_map[item.name])
                elif hasattr(item, "__dict__") or isinstance(item, list):
                    node[index] = self._replace_wildcards(item)
            return node

        if hasattr(node, "__dict__"):
            for attr, value in node.__dict__.items():
                if isinstance(value, WildcardExpression):
                    setattr(node, attr, copy.deepcopy(self.wildcard_expression_map[value.name]))
                elif isinstance(value, WildcardStatement):
                    setattr(node, attr, copy.deepcopy(self.wildcard_statement_map[value.name]))
                elif hasattr(value, "__dict__") or isinstance(value, list):
                    setattr(node, attr, self._replace_wildcards(value))
        return node


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

            if pattern.matches(node):
                pattern.apply(node)
                self._apply_program_meta(node, pattern)
        return node

    def _apply_program_meta(self, node: Program, pattern: Pattern):
        for name, value_expr in pattern.meta.items():
            if not hasattr(node, name):
                raise AttributeError(
                    f"{Path(pattern.filename).name}: meta field {name!r} does not exist on Program"
                )
            setattr(node, name, self._constant_fold_meta_value(value_expr, pattern, name))

    def _constant_fold_meta_value(self, expr: Expression, pattern: Pattern, name: str):
        try:
            return self._eval_constant_expr(expr)
        except ValueError as exc:
            raise ValueError(
                f"{Path(pattern.filename).name}: meta field {name!r} must be a constant expression"
            ) from exc

    def _eval_constant_expr(self, expr: Expression):
        if isinstance(expr, Number):
            return expr.value
        if isinstance(expr, UnaryMinusExpr):
            return -self._eval_constant_expr(expr.operand)
        if isinstance(expr, BinaryExpr):
            left = self._eval_constant_expr(expr.left)
            right = self._eval_constant_expr(expr.right)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                return left / right
            if expr.op == "%":
                return left % right
        raise ValueError
