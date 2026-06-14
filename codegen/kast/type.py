"""Type AST nodes for code generation."""

from codegen.kast.ast_node import AstNode
from codegen.kast.expression import Expression


class Type(AstNode):
    """Base class for type AST nodes."""

    def accept(self, visitor):
        return visitor.visit_type(self)


class Int(Type):
    """Represents 'int', 'size_t', or other integer types."""

    def __init__(self, name: str = "int"):
        self.name = name

    def accept(self, visitor):
        return visitor.visit_int(self)


class Float(Type):
    """Represents 'float' or 'rlmm_float' types."""

    def accept(self, visitor):
        return visitor.visit_float(self)


class Float16(Type):
    """Represents 'float16' or 'rlmm_float_small' types."""

    def accept(self, visitor):
        return visitor.visit_float16(self)


class FixedSizeVector(Type):
    """fixed_size_vector<elem_type, size_expr>&"""

    def __init__(self, elem_type: Type, size_expr: Expression):
        self.elem_type = elem_type
        self.size_expr = size_expr

    def accept(self, visitor):
        return visitor.visit_fixed_size_vector(self)


class FlexibleRowsMatrix(Type):
    """flexible_rows_matrix<elem_type, row_size_expr, col_size_expr>&"""

    def __init__(
        self, elem_type: Type, row_size_expr: Expression, col_size_expr: Expression
    ):
        self.elem_type = elem_type
        self.row_size_expr = row_size_expr
        self.col_size_expr = col_size_expr

    def accept(self, visitor):
        return visitor.visit_flexible_rows_matrix(self)


class FixedSizeMatrix(Type):
    """fixed_size_matrix<elem_type, row_size_expr, col_size_expr>&"""

    def __init__(
        self, elem_type: Type, row_size_expr: Expression, col_size_expr: Expression
    ):
        self.elem_type = elem_type
        self.row_size_expr = row_size_expr
        self.col_size_expr = col_size_expr

    def accept(self, visitor):
        return visitor.visit_fixed_size_matrix(self)


class FlexibleSizeMatrix(Type):
    """flexible_size_matrix<elem_type, row_size_expr, col_size_expr>&"""

    def __init__(
        self, elem_type: Type, row_size_expr: Expression, col_size_expr: Expression
    ):
        self.elem_type = elem_type
        self.row_size_expr = row_size_expr
        self.col_size_expr = col_size_expr

    def accept(self, visitor):
        return visitor.visit_flexible_size_matrix(self)


class FixedSizeObjVectorMatrix(Type):
    """fixed_size_obj_vector<matrix_type<elem, rows, cols>, levels>&"""

    def __init__(
        self,
        elem_type: Type,
        level_expr: Expression,
        row_size_expr: Expression,
        col_size_expr: Expression,
    ):
        self.elem_type = elem_type
        self.level_expr = level_expr
        self.row_size_expr = row_size_expr
        self.col_size_expr = col_size_expr

    def accept(self, visitor):
        return visitor.visit_fixed_size_obj_vector_matrix(self)


class FixedSizeLevelsRowsColsMatrix(Type):
    """fixed_size_levels_rows_cols_matrix<elem_type, level_expr, row_size_expr, col_size_expr>&"""

    def __init__(
        self,
        elem_type: Type,
        level_expr: Expression,
        row_size_expr: Expression,
        col_size_expr: Expression,
    ):
        self.elem_type = elem_type
        self.level_expr = level_expr
        self.row_size_expr = row_size_expr
        self.col_size_expr = col_size_expr

    def accept(self, visitor):
        return visitor.visit_fixed_size_levels_rows_cols_matrix(self)


class FlexibleRowsColsLevelsMatrix(Type):
    """flexible_rows_cols_levels_matrix<elem_type, level_expr, row_size_expr, col_size_expr>&"""

    def __init__(
        self,
        elem_type: Type,
        level_expr: Expression,
        row_size_expr: Expression,
        col_size_expr: Expression,
    ):
        self.elem_type = elem_type
        self.level_expr = level_expr
        self.row_size_expr = row_size_expr
        self.col_size_expr = col_size_expr

    def accept(self, visitor):
        return visitor.visit_flexible_rows_cols_levels_matrix(self)


__all__ = [
    "Type",
    "Int",
    "Float",
    "Float16",
    "FixedSizeVector",
    "FlexibleRowsMatrix",
    "FixedSizeMatrix",
    "FixedSizeLevelsRowsColsMatrix",
    "FlexibleRowsColsLevelsMatrix",
    "FlexibleRowsColsMatrix",
    "FlexibleSizeMatrix",
    "FixedSizeObjVectorMatrix",
]


class FlexibleRowsColsMatrix(Type):
    """flexible_rows_cols_matrix<elem_type, row_size_expr, col_size_expr>&"""

    def __init__(
        self,
        elem_type: Type,
        row_size_expr: Expression,
        col_size_expr: Expression,
    ):
        self.elem_type = elem_type
        self.row_size_expr = row_size_expr
        self.col_size_expr = col_size_expr

    def accept(self, visitor):
        return visitor.visit_flexible_rows_cols_matrix(self)
