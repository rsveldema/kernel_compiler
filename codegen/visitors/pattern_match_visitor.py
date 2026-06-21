from codegen.kast.expression import (
    ArrayAccess,
    BinaryExpr,
    CallExpr,
    Expression,
    FieldAccess,
    Identifier,
    LimitExpr,
    NegationExpr,
    Number,
    TernaryExpr,
    UnaryMinusExpr,
)
from codegen.kast.program import Program
from codegen.kast.statement import (
    Assignment,
    AtomicOp,
    CallStatement,
    Condition,
    Declaration,
    ForLoopRange,
    ForLoopWithConditionAndIncrement,
    If,
    OverflowCheck,
    RawStatement,
    ReturnStatement,
    SharedDecl,
    Statement,
    TensorLayoutDecl,
)
from codegen.kast.type import Float, Int
from codegen.visitors import visitor


class PatternMatchVisitor(visitor.Visitor):
    def __init__(self):
        self.wildcard_statement_map: dict[str, Statement] = {}
        self.wildcard_expression_map: dict[str, Expression] = {}
        self._candidate_stack = []

    def matches_statements(self, patterns: list[Statement], statements: list[Statement]) -> bool:
        if not patterns or len(patterns) != len(statements):
            return False

        for pattern, statement in zip(patterns, statements):
            if not self._match(pattern, statement):
                return False

        return True

    def _match(self, pattern, candidate) -> bool:
        if pattern is None or candidate is None:
            return pattern is candidate

        self._candidate_stack.append(candidate)
        try:
            return pattern.accept(self)
        finally:
            self._candidate_stack.pop()

    def _candidate(self):
        return self._candidate_stack[-1]

    def _same_node(self, left, right) -> bool:
        return PatternMatchVisitor()._match(left, right)

    def _bind_statement(self, name: str, statement: Statement) -> bool:
        bound = self.wildcard_statement_map.get(name)
        if bound is None:
            self.wildcard_statement_map[name] = statement
            return True
        return self._same_node(bound, statement)

    def _bind_expression(self, name: str, expression: Expression) -> bool:
        bound = self.wildcard_expression_map.get(name)
        if bound is None:
            self.wildcard_expression_map[name] = expression
            return True
        return self._same_node(bound, expression)

    def _match_type(self, pattern, attrs=()) -> bool:
        candidate = self._candidate()
        if type(candidate) is not type(pattern):
            return False
        return all(self._match(getattr(pattern, attr), getattr(candidate, attr)) for attr in attrs)

    def _match_statement_list(self, patterns, statements) -> bool:
        if len(patterns) != len(statements):
            return False
        return all(self._match(pattern, statement) for pattern, statement in zip(patterns, statements))

    def _match_expr_list(self, patterns, expressions) -> bool:
        if len(patterns) != len(expressions):
            return False
        return all(self._match(pattern, expression) for pattern, expression in zip(patterns, expressions))

    def visit_type(self, node):
        return type(self._candidate()) is type(node)

    def visit_int(self, node):
        candidate = self._candidate()
        return isinstance(candidate, Int) and candidate.name == node.name

    def visit_float(self, node):
        return isinstance(self._candidate(), Float)

    def visit_float16(self, node):
        return type(self._candidate()) is type(node)

    def visit_coop_mat(self, node):
        return self._match_type(
            node,
            ("elem_type", "scope_expr", "row_size_expr", "col_size_expr", "use_expr"),
        )

    def visit_fixed_size_vector(self, node):
        return self._match_type(node, ("elem_type", "size_expr"))

    def visit_flexible_rows_matrix(self, node):
        return self._match_type(node, ("elem_type", "row_size_expr", "col_size_expr"))

    def visit_fixed_size_matrix(self, node):
        return self._match_type(node, ("elem_type", "row_size_expr", "col_size_expr"))

    def visit_fixed_size_triangular_matrix(self, node):
        return self._match_type(node, ("elem_type", "row_size_expr", "col_size_expr"))

    def visit_flexible_size_matrix(self, node):
        return self._match_type(node, ("elem_type", "row_size_expr", "col_size_expr"))

    def visit_fixed_size_obj_vector_matrix(self, node):
        return self._match_type(
            node,
            ("elem_type", "level_expr", "row_size_expr", "col_size_expr"),
        )

    def visit_fixed_size_levels_rows_cols_matrix(self, node):
        return self._match_type(
            node,
            ("elem_type", "level_expr", "row_size_expr", "col_size_expr"),
        )

    def visit_flexible_rows_cols_levels_matrix(self, node):
        return self._match_type(
            node,
            ("elem_type", "level_expr", "row_size_expr", "col_size_expr"),
        )

    def visit_flexible_rows_cols_matrix(self, node):
        return self._match_type(node, ("elem_type", "row_size_expr", "col_size_expr"))

    def visit_tensor_layout(self, node):
        return self._match_type(node, ("dim_expr",))

    def visit_expression(self, node):
        return type(self._candidate()) is type(node)

    def visit_number(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, Number)
            and candidate.value == node.value
            and candidate.unsigned == node.unsigned
        )

    def visit_identifier(self, node):
        candidate = self._candidate()
        return isinstance(candidate, Identifier) and candidate.name == node.name

    def visit_array_access(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, ArrayAccess)
            and self._match(node.base, candidate.base)
            and self._match_expr_list(node.indices, candidate.indices)
        )

    def visit_field_access(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, FieldAccess)
            and node.field == candidate.field
            and self._match(node.base, candidate.base)
        )

    def visit_limit_expr(self, node):
        return self._match_type(node, ("max_val", "start", "end"))

    def visit_binary_expr(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, BinaryExpr)
            and node.op == candidate.op
            and self._match(node.left, candidate.left)
            and self._match(node.right, candidate.right)
        )

    def visit_call_expr(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, CallExpr)
            and self._match(node.callee, candidate.callee)
            and self._match_expr_list(node.args, candidate.args)
        )

    def visit_cast_expr(self, node):
        return self._match_type(node, ("cast_type", "operand"))

    def visit_negation_expr(self, node):
        candidate = self._candidate()
        return isinstance(candidate, NegationExpr) and self._match(node.operand, candidate.operand)

    def visit_wildcard_expression(self, node):
        candidate = self._candidate()
        return isinstance(candidate, Expression) and self._bind_expression(node.name, candidate)

    def visit_ternary_expr(self, node):
        return self._match_type(node, ("condition", "true_expr", "false_expr"))

    def visit_unary_minus_expr(self, node):
        candidate = self._candidate()
        return isinstance(candidate, UnaryMinusExpr) and self._match(node.operand, candidate.operand)

    def visit_condition(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, Condition)
            and node.op == candidate.op
            and self._match(node.lhs, candidate.lhs)
            and self._match(node.rhs, candidate.rhs)
        )

    def visit_statement(self, node):
        return type(self._candidate()) is type(node)

    def visit_for_loop_range(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, ForLoopRange)
            and node.loop_var_name == candidate.loop_var_name
            and self._match(node.loop_var_type, candidate.loop_var_type)
            and self._match(node.init_expr, candidate.init_expr)
            and self._match_statement_list(node.body_stmts, candidate.body_stmts)
        )

    def visit_for_loop_with_condition_and_increment(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, ForLoopWithConditionAndIncrement)
            and node.loop_var_name == candidate.loop_var_name
            and node.increment_var == candidate.increment_var
            and node.increment_op == candidate.increment_op
            and self._match(node.loop_var_type, candidate.loop_var_type)
            and self._match(node.init_expr, candidate.init_expr)
            and self._match(node.condition, candidate.condition)
            and self._match_statement_list(node.body_stmts, candidate.body_stmts)
        )

    def visit_if(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, If)
            and self._match(node.condition, candidate.condition)
            and self._match_statement_list(node.body_stmts, candidate.body_stmts)
            and self._match_statement_list(node.else_stmts, candidate.else_stmts)
        )

    def visit_declaration(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, Declaration)
            and node.is_const == candidate.is_const
            and node.is_constexpr == candidate.is_constexpr
            and node.name == candidate.name
            and self._match(node.var_type, candidate.var_type)
            and self._match(node.init_expr, candidate.init_expr)
        )

    def visit_assignment(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, Assignment)
            and node.assign_op == candidate.assign_op
            and self._match(node.lvalue, candidate.lvalue)
            and self._match(node.rvalue, candidate.rvalue)
        )

    def visit_overflow_check(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, OverflowCheck)
            and self._match(node.lvalue, candidate.lvalue)
            and self._match(node.operand, candidate.operand)
        )

    def visit_shared_decl(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, SharedDecl)
            and node.is_const == candidate.is_const
            and node.is_constexpr == candidate.is_constexpr
            and node.name == candidate.name
            and self._match(node.var_type, candidate.var_type)
            and self._match(node.init_expr, candidate.init_expr)
            and self._match_expr_list(node.dimensions, candidate.dimensions)
        )

    def visit_raw_statement(self, node):
        candidate = self._candidate()
        return isinstance(candidate, RawStatement) and candidate.text == node.text

    def visit_call_statement(self, node):
        candidate = self._candidate()
        return isinstance(candidate, CallStatement) and self._match(node.call_expr, candidate.call_expr)

    def visit_wildcard_statement(self, node):
        candidate = self._candidate()
        return isinstance(candidate, Statement) and self._bind_statement(node.name, candidate)

    def visit_workgroup_properties(self, node):
        return self._match_type(node, ("x_expr", "y_expr", "z_expr"))

    def visit_program(self, node):
        candidate = self._candidate()
        return isinstance(candidate, Program) and self._match_statement_list(node.body_stmts, candidate.body_stmts)

    def visit_return_statement(self, node):
        return isinstance(self._candidate(), ReturnStatement)

    def visit_atomic_op(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, AtomicOp)
            and node.op == candidate.op
            and self._match(node.lhs, candidate.lhs)
            and self._match(node.rhs, candidate.rhs)
        )

    def visit_tensor_layout_decl(self, node):
        candidate = self._candidate()
        return (
            isinstance(candidate, TensorLayoutDecl)
            and node.name == candidate.name
            and self._match(node.dim_expr, candidate.dim_expr)
            and self._match(node.init_expr, candidate.init_expr)
        )
