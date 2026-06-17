
from kernel_compiler.codegen.kast.program import Program
from kernel_compiler.codegen.visitors import visitor

class Pattern:
    def __init__(self, filename: str):
        self.filename = filename
        searh_pattern = parse(filename)


    def matches(self, node):
        # Placeholder for pattern matching logic
        return False

class PatternStore:
    def __init__(self):
        self.patterns = {}


class TreeRewriter(visitor.Visitor):
    def __init__(self, context):
        self.context = context


    def visit_program(self, node: Program):
        return super().visit_program(node)
    