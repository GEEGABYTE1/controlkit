#backend neural int. representation for control laws

from __future__ import annotations
from dataclasses import dataclass, field 
from enum import StrEnum 
from typing import TYPE_CHECKING, Mapping


if TYPE_CHECKING:
    from controlkit.policies.base import PolicyKind 

class IRValidationError(ValueError):
    # Raised when an IR node is malformed or shape-incompatible.
    pass 


class ValueKind(StrEnum):

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

class Expr:
    @property 
    def shape(self) -> Shape:
        raise notImplementedError("Subclasses must implement shape property")
    
@dataclass(frozen=True)
class ScalarConstant(Expr):
    value: float 
    @property 
    def shape(self) -> Shape:
        return Shape.scalar() 
    
    def __repr__(self) -> str: 
        return f"const({self.value:g})"
    
@dataclass(frozen=True)
class Vector(Expr):
    name: str 
    dim: int 
    def __post_init__(self) -> None:
        _validate_name(self.name)
        if self.dim <= 0:
            raise IRValidationError(f"Vector dimension must be positive: {self.dim}")
        
    @property 
    def shape(self) -> Shape:
        return Shape.vector(self.dim)
    
    def __repr__(self) -> str:
        return f"{self.name}:vector[{self.dim}]" 
    
@dataclass(frozen=True)
class Matrix(Expr):
    name:str 
    rows:int 
    cols:int 

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if self.rows <= 0 or self.cols <= 0:
            raise IRValidationError(f"Matrix dimensions must be positive: {self.rows}x{self.cols}")
    
    @property
    def shape(self) -> Shape:
        return Shape.matrix(self.rows, self.cols) 
    
    def __repr__(self) -> str:
        return f"{self.name}:matrix[{self.rows}x{self.cols}]" 
    
@dataclass(frozen=True)
class Neg(Expr):
    #negation 
    value: Expr 
    def __post_init__(self) -> None:
        _require_expr(self.value, "value")

        
    @property 
    def shape(self) -> Shape:
        return self.value.shape
    
    def __repr__(self) -> str:
        return f"neg({self.value})"
    
@dataclass(frozen=True)
class MatVecMul(Expr):
    #matrix vec mult 
    matrix: Expr 
    vector: Expr 

    def __post_init__(self) -> None:
        _require_expr(self.matrix, "matrix")
        _require_expr(self.vector, "vector")
        if self.matrix.shape.kind is not ValueKind.MATRIX:
            raise IRValidationError(f"MatVecMul requires matrix operand: {self.matrix}") 
        if self.vector.shape.kind is not ValueKind.VECTOR:
            raise IRValidationError(f"MatVecMul requires vector operand: {self.vector}")
        if self.matrix.shape.cols != self.vector.shape.rows:
            raise IRValidationError(f"MatVecMul shape mismatch: matrix cols {self.matrix.shape.cols} != vector rows {self.vector.shape.rows}")
    
    @property 
    def shape(self) -> Shape:
        return Shape.vector(self.matrix.shape.rows) 

    def __repr__(self) -> str: 
        return f"matvec({self.matrix}, {self.vector})"

    
@dataclass(frozen=True)
class Add(Expr):
    #element-wise mult 
    left: Expr 
    right: Expr 

    def __post_init__(self) -> None:
        _validate_same_shape(self.left, self.right, "addition")

    @property 
    def shape(self) -> Shape:
        return self.left.shape 

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"
    
@dataclass(frozen=True)
class Sub(Expr): 
    "element-wise subtraction"
    left: Expr 
    right: Expr

    def __post_init__(self) -> None: 
        _validate_same_shape(self.left, self.right, "subtraction") 
    
    @property 
    def shape(self) -> Shape: 
        return self.left.shape 
    
    def __repr__(self) -> str:
        return f"({self.left} - {self.right})" 
    
@dataclass(frozen=True)
class Clip(Expr):
    #element-wise clipping/saturation

    value: Expr 
    lower: Expr 
    upper: Expr 

    def __post_init__(self) -> None: 
        _require_expr(self.value, "value")
        _require_expr(self.lower, "lower")
        _require_expr(self.upper, "upper")
        if self.value.shape != self.lower.shape or self.value.shape != self.upper.shape:
            raise IRValidationError(f"Clip requires value, lower, and upper to have the same shape: {self.value.shape}, {self.lower.shape}, {self.upper.shape}") 
        
    @property 
    def shape(self) -> Shape:  
        return self.value.shape

    def __repr__(self) -> str: 
        return f"(clip {self.value} to [{self.lower}, {self.upper}])"
    

class DynamicsKind(StrEnum):
    #linear-system dynamics forms 
    CONTINUOUS = "continuous" 
    DISCRETE = "discrete"

@dataclass(frozen=True)
class LinearSystemIR:
    # Linear system `x_dot = Ax + Bu` or `x_next = Ax + Bu`
    state: Vector 
    control: Vector 
    a_matrix: Matrix 
    b_matrix: Matrix 
    dynamics: DynamicsKind 

    def __post_init__(self) -> None:
        if not isinstance(self.state, Vector):
            raise IRValidationError(f"State must be a vector: {self.state}") 
        if not isinstance(self.control, Vector):
            raise IRValidationError(f"Control must be a vector: {self.control}")
        if not isinstance(self.a_matrix, Matrix):
            raise IRValidationError(f"A matrix must be a matrix: {self.a_matrix}")
        if not isinstance(self.b_matrix, Matrix):
            raise IRValidationError(f"B matrix must be a matrix: {self.b_matrix}")
        
        state_dim = self.state.dim 
        control_dim = self.control.dim
        if (self.a_matrix.rows, self.a_matrix.cols) != (state_dim, state_dim):
            raise IRValidationError(f"A matrix shape must be ({state_dim}x{state_dim}): {self.a_matrix.shape}") 
        if (self.b_matrix.rows, self.b_matrix.cols) != (state_dim, control_dim):
            raise IRValidationError(f"B matrix shape must be ({state_dim}x{control_dim}): {self.b_matrix.shape}")   
        
        @property 
        def expression(self) -> Add:
            return Add(MatVecMul(self.a_matrix, self.state), MatVecMul(self.b_matrix, self.control)) 
        
        @property 
        def lhs_name(self) -> str:
            return "x_dot" if self.dynamics == DynamicsKind.CONTINUOUS else "x_next"
        
        def __repr__(self) -> str:
            return f"{self.lhs_name} = {self.expression}" 
        
    
@dataclass(frozen=True)
class IRModule:
    # a policy for backend-neutral representation
    name: str 
    policy: PolicyKind
    metadata: Mapping[str, str] = field(default_factory=dict)
    systems: tuple[LinearSystemIR, ...] = field(default_factory=tuple)  
    control_laws: tuple[Expr, ...] = field(default_factory=tuple) 

    def __repr__(self) -> str:
        policy_name = getattar(self.policy, "value", str(self.policy))
        return f"IRModule(name={self.name}, policy={policy_name}, systems={self.systems}, control_laws={self.control_laws})"    
    
def matvec(matrix: Expr, vector: Expr) -> MatVecMul:
    return MatVecMul(matrix, vector)

def add(left: Expr, right: Expr) -> Add: 
    return Add(left=left, right=right)

def sub(left: Expr, right: Expr) -> Sub:
    return Sub(left=left, right=right) 

def neg(value: Expr) -> Neg:
    return Neg(value=value) 

def clip(value: Expr, lower: Expr | float, upper: Expr | float) -> Clip:
    return Clip(value=value, lower=_coerce_bound(lower), upper=_coerce_bound(upper))


def _coerce_bound(bound: Expr | float) -> Expr:
    if isinstance(bound, (int, float)):
        return ScalarConstant(float(bound))
    _require_expr(bound, "bound")
    return bound 

def _require_expr(value: object, name: str) -> None:
    if not isinstance(value, Expr):
        raise TypeError(f"{name} must be an Expr: got {type(value).__name__}") 

def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string: got {type(name).__name__}") 
    if not name:
        raise IRValidationError("Name cannot be empty") 

def _validate_same_shape(left: Expr, right: Expr, operation: str) -> None:
    _require_expr(left, "left")
    _require_expr(right, "right")
    if left.shape != right.shape:
        raise IRValidationError(f"{operation.capitalize()} requires operands to have the same shape: {left.shape} vs {right.shape}")
    

def _validate_bound(value: Expr, bound: Expr, name:str) -> None:
    _require_expr(bound, name)
    if bound.shape.kind is ValueKind.SCALAR:
        return 

    if bound.shape != value.shape:
        raise IRValidationError(f"Bound '{name}' must be either scalar or have the same shape as value: {bound.shape} vs {value.shape}") 
    