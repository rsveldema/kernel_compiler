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
import logging
from codegen.visitors.pattern_match_visitor import PatternMatchVisitor

logger = logging.getLogger("tree_rewriter")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    _fh = logging.FileHandler("tree-rewrite.log", mode="w")
    _fh.setLevel(logging.DEBUG)
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)


def _stmt_label(stmt: Statement) -> str:
    """Generate a human-readable label for a statement for logging."""
    if stmt is None:
        return "(None)"
    t = type(stmt).__name__
    if hasattr(stmt, 'body_stmts') and stmt.body_stmts:
        return f"{t}[{len(stmt.body_stmts)} stmts]"
    if hasattr(stmt, 'body') and stmt.body:
        return f"{t}[{len(stmt.body)} stmts]"
    if hasattr(stmt, 'condition'):
        return f"{t}({stmt.condition})"
    if hasattr(stmt, 'lhs'):
        lhs = stmt.lhs
        if hasattr(lhs, 'name'):
            return f"{t}(lhs={lhs.name})"
        return f"{t}(lhs={lhs})"
    if hasattr(stmt, 'expression'):
        return f"{t}(expr={stmt.expression})"
    return t


def _expr_label(expr: Expression) -> str:
    """Generate a human-readable label for an expression for logging."""
    if expr is None:
        return "(None)"
    if isinstance(expr, WildcardExpression):
        return f"wildcard({expr.name})"
    if isinstance(expr, Number):
        return str(expr.value)
    if isinstance(expr, Identifier):
        return expr.name
    if isinstance(expr, BinaryExpr):
        return f"({_expr_label(expr.left)} {expr.op} {_expr_label(expr.right)})"
    if isinstance(expr, UnaryMinusExpr):
        return f"(-{_expr_label(expr.operand)})"
    return type(expr).__name__


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
        pattern_name = Path(self.filename).stem
        logger.info("=== Loading pattern file: %s ===", self.filename)
        logger.info("  optimizes header: %s", self.target_header)
        logger.info("  raw meta text keys found: %s",
                     re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=', text[text.rfind("meta"):]))

        try:
            pattern_tree = parse_search_replace_pattern(text)
        except Exception as exc:
            logger.error("  PARSE FAILED for %s: %s", pattern_name, exc)
            raise RuntimeError(f"Pattern file {self.filename} could not be parsed: {exc}") from exc

        logger.info("  Parse succeeded for %s", pattern_name)

        self.search.clear()
        self.replace.clear()

        for child in pattern_tree.children:
            if not isinstance(child, Tree):
                logger.debug("  Skipping non-Tree child: %s", type(child).__name__)
                continue

            if child.data == "search_statements":
                self.search.extend(self._transform_statements(child))
                logger.info("  Parsed %d search statement(s):", len(self.search))
                for i, s in enumerate(self.search):
                    logger.info("    [%d] %s", i, _stmt_label(s))
            elif child.data == "replace_statements":
                self.replace.extend(self._transform_statements(child))
                logger.info("  Parsed %d replace statement(s):", len(self.replace))
                for i, s in enumerate(self.replace):
                    logger.info("    [%d] %s", i, _stmt_label(s))
            elif child.data == "constraints":
                self.constraints = self._transform_constraints(child)
                logger.info("  Parsed constraint: %s", _expr_label(self.constraints))
            elif child.data == "meta":
                self.meta.update(self._transform_meta(child))
                resolved = {k: _expr_label(v) for k, v in self.meta.items()}
                logger.info("  Parsed meta keys: %s -> %s",
                             list(self.meta.keys()), resolved)
            else:
                logger.debug("  Unknown tree child data: %s", child.data)

        if not self.search and not self.replace:
            logger.error("  Pattern '%s' has NO search or replace statements after parsing!", pattern_name)
            raise RuntimeError(
                f"Pattern file {self.filename} has no search or replace statements after parsing. "
                f"Check the .tkernel syntax."
            )

        logger.info("  Pattern %s loaded: search=%d, replace=%d, meta=%s",
                     pattern_name, len(self.search), len(self.replace),
                     list(self.meta.keys()))
        if self.constraints:
            logger.info("  Pattern %s constraint: %s",
                         pattern_name, _expr_label(self.constraints))

    def _load_meta_from_text(self, text: str) -> None:
        match = re.search(r'\boptimizes\s*\(\s*"([^"]+)"\s*\)', text)
        if match:
            self.target_header = match.group(1)

        meta_text = self._extract_meta_block(text)
        if meta_text is None:
            logger.debug("  No meta block found in text")
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
                logger.debug("  Meta assignment: %s = %r (raw: %s)", name, _expr_label(value), raw_value)
            else:
                logger.warning("  Meta assignment FAILED to parse: %s = %r", name, raw_value)

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
                    logger.debug("    Transformed statement: %s", _stmt_label(statement))
            elif isinstance(child, Tree):
                statement = transform_statement(child)
                if statement is not None:
                    statements.append(statement)
                    logger.debug("    Transformed statement (non-standard): %s", _stmt_label(statement))
            else:
                logger.debug("    Skipping non-Tree child in statement list: %s", type(child).__name__)

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
        pattern_name = Path(self.filename).stem
        logger.info("--- Pattern '%s' matching for header '%s' ---", pattern_name, node.header)

        if not self.search:
            logger.warning("Pattern '%s': search list EMPTY — cannot match!", pattern_name)
            self.wildcard_statement_map.clear()
            self.wildcard_expression_map.clear()
            return False

        if len(self.search) != len(node.body_stmts):
            logger.info("Pattern '%s': search count %d != body_stmts count %d, no match",
                         pattern_name, len(self.search), len(node.body_stmts))
            logger.info("  Search stmt types: %s",
                         [_stmt_label(s) for s in self.search])
            logger.info("  Body stmt types: %s",
                         [_stmt_label(s) for s in node.body_stmts])
            self.wildcard_statement_map.clear()
            self.wildcard_expression_map.clear()
            return False

        logger.info("Pattern '%s': checking %d statements against %d body statements...",
                     pattern_name, len(self.search), len(node.body_stmts))

        matcher = PatternMatchVisitor()
        match_result = matcher.matches_statements(self.search, node.body_stmts)

        if not match_result:
            logger.info("Pattern '%s': NO MATCH", pattern_name)
            logger.info("  Final wildcard bindings: exprs=%s, stmts=%s",
                         list(matcher.wildcard_expression_map.keys()),
                         list(matcher.wildcard_statement_map.keys()))
            self.wildcard_statement_map.clear()
            self.wildcard_expression_map.clear()
            return False

        self.wildcard_statement_map = matcher.wildcard_statement_map
        self.wildcard_expression_map = matcher.wildcard_expression_map

        logger.info("Pattern '%s': MATCHED!", pattern_name)
        logger.info("  Wildcard expression bindings:")
        for k, v in self.wildcard_expression_map.items():
            logger.info("    %s -> %s", k, _expr_label(v))
        logger.info("  Wildcard statement bindings:")
        for k, v in self.wildcard_statement_map.items():
            logger.info("    %s -> %s", k, _stmt_label(v))

        if self.constraints:
            # Evaluate constraints against the program (node)
            try:
                constraint_val = self._eval_constraint(self.constraints, node)
                logger.info("  Constraint result: %s = %s",
                             _expr_label(self.constraints), constraint_val)
                if not constraint_val:
                    logger.info("  Constraint evaluated to FALSE — no match")
                    self.wildcard_statement_map.clear()
                    self.wildcard_expression_map.clear()
                    return False
            except Exception as exc:
                logger.warning("  Constraint evaluation threw exception: %s", exc)
                logger.info("  Constraint evaluated with exception — skipping constraint")

        logger.info("Pattern '%s' matched — applying replacement...", pattern_name)
        return True

    def _eval_constraint(self, constraint: Expression, node: Program) -> bool:
        """Evaluate a constraint expression against the program state."""
        val = self._eval_constant_expr(constraint)
        return bool(val)

    def apply(self, node: Program):
        logger.info("--- Pattern '%s' APPLYING replacement ---", Path(self.filename).stem)
        logger.info("  Replace list has %d statement(s):", len(self.replace))
        for i, stmt in enumerate(self.replace):
            logger.info("    [%d] %s", i, _stmt_label(stmt))

        if not self.replace:
            logger.error("  REPLACE LIST IS EMPTY! Pattern matched but has no replacement statements.")
            raise RuntimeError(
                f"Pattern {Path(self.filename).name} matched but has no replacement statements. "
                f"This is a bug — the pattern matched but cannot produce output."
            )

        old_count = len(node.body_stmts)
        node.body_stmts = [self._clone_statement(statement) for statement in self.replace]
        new_count = len(node.body_stmts)

        logger.info("  Replaced %d statements with %d statement(s)", old_count, new_count)
        logger.info("  New body statements:")
        for i, stmt in enumerate(node.body_stmts):
            logger.info("    [%d] %s", i, _stmt_label(stmt))

        print(f"TREE TRANSFORMED: {node.header}")
        node.tree_transformed = True

    def _clone_statement(self, statement: Statement) -> Statement:
        if isinstance(statement, WildcardStatement):
            result = copy.deepcopy(self.wildcard_statement_map[statement.name])
            logger.debug("    WildcardStmt '%s' -> %s", statement.name, _stmt_label(result))
            return result
        return self._replace_wildcards(copy.deepcopy(statement))

    def _replace_wildcards(self, node):
        if isinstance(node, WildcardExpression):
            if node.name in self.wildcard_expression_map:
                result = copy.deepcopy(self.wildcard_expression_map[node.name])
                logger.debug("      WildcardExpr '%s' -> %s", node.name, _expr_label(result))
                return self._replace_wildcards(result)
            if node.name in self.meta:
                result = copy.deepcopy(self.meta[node.name])
                logger.debug("      Meta '%s' -> %s", node.name, _expr_label(result))
                return self._replace_wildcards(result)
            logger.warning("      UNRESOLVED WildcardExpr '%s'!", node.name)
            return copy.deepcopy(node)

        if isinstance(node, list):
            for index, item in enumerate(node):
                if isinstance(item, WildcardExpression):
                    node[index] = self._replace_wildcards(item)
                elif isinstance(item, WildcardStatement):
                    node[index] = copy.deepcopy(self.wildcard_statement_map[item.name])
                elif hasattr(item, "__dict__") or isinstance(item, list):
                    node[index] = self._replace_wildcards(item)
            return node

        if hasattr(node, "__dict__"):
            for attr, value in node.__dict__.items():
                if isinstance(value, WildcardExpression):
                    if value.name in self.wildcard_expression_map:
                        setattr(node, attr, self._replace_wildcards(copy.deepcopy(self.wildcard_expression_map[value.name])))
                    elif value.name in self.meta:
                        setattr(node, attr, self._replace_wildcards(copy.deepcopy(self.meta[value.name])))
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

        logger.info("=== PatternStore.init: Scanning transforms directory: %s ===", transforms_dir)
        tkernel_files = list(transforms_dir.glob("*.tkernel"))
        logger.info("Found %d .tkernel file(s): %s", len(tkernel_files), [f.name for f in tkernel_files])

        for pattern_path in sorted(tkernel_files):
            logger.info("--- Loading pattern: %s ---", pattern_path.name)
            try:
                pattern = Pattern(str(pattern_path))
                has_content = len(pattern.search) > 0 or len(pattern.replace) > 0
                status = "OK" if has_content else "EMPTY (parse or init issue)"
                logger.info("  -> Pattern '%s': search=%d, replace=%d [%s]",
                             pattern_path.stem, len(pattern.search), len(pattern.replace), status)
                if not has_content:
                    logger.error("  -> Pattern '%s' has NO content! Skipping.", pattern_path.stem)
                self.patterns[pattern_path.stem] = pattern
            except Exception as e:
                logger.error("  -> EXCEPTION loading %s: %s", pattern_path.name, e)
                print(f"Skipping unparseable pattern {pattern_path.name}: {e}")

        logger.info("=== Loaded %d pattern(s) into store: %s ===",
                     len(self.patterns), list(self.patterns.keys()))


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
        logger.info("=== TreeRewriter.visit_program: %s (%d body statements) ===",
                     node.header, len(node.body_stmts))
        logger.info("  Body statement types: %s", [_stmt_label(s) for s in node.body_stmts])
        logger.info("  Available patterns: %s", list(self.pattern_store.patterns.keys()))

        applied_any = False
        for pname, pattern in self.pattern_store.patterns.items():
            logger.info("--- Trying pattern '%s' ---", pname)
            logger.info("  Pattern: search=%d, replace=%d", len(pattern.search), len(pattern.replace))
            if pattern.constraints:
                logger.info("  Pattern has constraint: %s", _expr_label(pattern.constraints))
            else:
                logger.info("  Pattern has NO constraint")

            if pattern.matches(node):
                logger.info("Pattern '%s' matched — applying replacement...", pname)
                try:
                    pattern.apply(node)
                    self._apply_program_meta(node, pattern)
                    logger.info("  -> Program transformed: %d -> %d statements",
                                 len(body_stmts), len(node.body_stmts))
                    applied_any = True
                except Exception as exc:
                    logger.error("  -> FAILED to apply pattern '%s': %s", pname, exc)
                    raise RuntimeError(
                        f"Pattern '{pname}' matched for header '{node.header}' "
                        f"but replacement failed: {exc}"
                    ) from exc
            else:
                logger.info("Pattern '%s' did not match, continuing", pname)

        if not applied_any:
            logger.info("  No patterns applied to '%s'", node.header)

        logger.info("=== visit_program done for %s (transformed=%s) ===",
                     node.header, getattr(node, 'tree_transformed', False))
        return node

    def _apply_program_meta(self, node: Program, pattern: Pattern):
        for name, value_expr in pattern.meta.items():
            if not hasattr(node, name):
                raise AttributeError(
                    f"{Path(pattern.filename).name}: meta field {name!r} does not exist on Program"
                )
            old_val = getattr(node, name, "<MISSING>")
            setattr(node, name, self._constant_fold_meta_value(value_expr, pattern, name))
            logger.info("  Meta set: %s = %s (was %s)", name, getattr(node, name), old_val)

    def _constant_fold_meta_value(self, expr: Expression, pattern: Pattern, name: str):
        try:
            return self._eval_constant_expr(expr, pattern, resolving={name})
        except ValueError as exc:
            raise ValueError(
                f"{Path(pattern.filename).name}: meta field {name!r} must be a constant expression"
            ) from exc

    def _eval_constant_expr(self, expr: Expression, pattern: Pattern | None = None, resolving: set[str] | None = None):
        if isinstance(expr, Number):
            return expr.value
        if isinstance(expr, WildcardExpression):
            if pattern is not None and expr.name in pattern.wildcard_expression_map:
                return self._eval_constant_expr(pattern.wildcard_expression_map[expr.name], pattern, resolving)
            if pattern is not None and expr.name in pattern.meta:
                if resolving is not None and expr.name in resolving:
                    raise ValueError
                next_resolving = set(resolving or set())
                next_resolving.add(expr.name)
                return self._eval_constant_expr(pattern.meta[expr.name], pattern, next_resolving)
            raise ValueError
        if isinstance(expr, Identifier):
            if pattern is not None and expr.name in pattern.meta:
                if resolving is not None and expr.name in resolving:
                    raise ValueError
                next_resolving = set(resolving or set())
                next_resolving.add(expr.name)
                return self._eval_constant_expr(pattern.meta[expr.name], pattern, next_resolving)
            raise ValueError
        if isinstance(expr, UnaryMinusExpr):
            return -self._eval_constant_expr(expr.operand, pattern, resolving)
        if isinstance(expr, BinaryExpr):
            left = self._eval_constant_expr(expr.left, pattern, resolving)
            right = self._eval_constant_expr(expr.right, pattern, resolving)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                if isinstance(left, int) and isinstance(right, int) and right != 0 and left % right == 0:
                    return left // right
                return left / right
            if expr.op == "%":
                return left % right
        raise ValueError
