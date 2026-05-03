#backend neural int. representation for control laws

from __future__ import annotations
from dataclasses import dataclass, field 
from enum import StrEnum 
from typing import TYPE_CHECKING, Mapping


if TYPE_CHECKING:
    from controlkit.policies.base import PolicyKind 

class IRValidationError(ValueError):
    """Raised when an IR node is malformed or shape-incompatible."""


class ValueKind(StrEnum):
    """Kinds of values expressible in ControlKit IR."""

    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass(frozen=True)
class Shape:
    #constant shape for ir value
    kind: ValueKind 
    rows: int = 1
    cols: int = 1 

    def __post_iniT__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise IRValidationError(f"Invalid shape with non-positive dimensions: {self}")
        if self.kind is ValueKind.SCALAR and (self.rows != 1 or self.cols != 1):
            raise IRValidationError(f"Scalar shape must be 1x1: {self}")
        if self.kind is ValueKind.VECTOR and self.cols != 1:
            raise IRValidationError(f"Vector shape must have one column: {self}")

        @classmethod
        def scalar(cls) -> Shape:
            return cls(ValueKind.SCALAR)   
         
        @classmethod 
        def vector(cls, size: int) -> Shape:
            return cls(ValueKind.VECTOR, rows=size)
        
        @classmethod 
        def matrix(cls, rows:int, cols:int) -> Shape:
            return cls(ValueKind.MATRIX, rows=rows, cols=cols) 
        
        @property
        def dim(self) -> int:
            if self.kind is not ValueKind.VECTOR:
                raise IRValidationError(f"Only vectors have a dimension: {self}")
            return self.rows 
        
        def __repr__(self) -> str:
            if self.kind is ValueKind.SCALAR:
                return "Scalar"
            elif self.kind is ValueKind.VECTOR:
                return f"Vector({self.rows})"
            elif self.kind is ValueKind.MATRIX:
                return f"Matrix({self.rows}x{self.cols})"
            else:
                return f"UnknownShape(kind={self.kind}, rows={self.rows}, cols={self.cols})"


        
