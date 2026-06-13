from abc import ABC, abstractmethod

from codegen.visitors.visitor import Visitor


class AstNode(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def accept(self, visitor: Visitor):
        pass
