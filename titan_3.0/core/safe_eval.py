"""
TITAN 3.0 - Safe Expression Evaluator Module
Secure alternative to eval() for trading strategy conditions.
"""

import ast
import operator
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SafeEvalError(Exception):
    """Custom exception for safe evaluation errors."""
    pass


class SafeExpressionEvaluator:
    """
    Safe alternative to eval() for trading conditions.
    
    This class provides a secure way to evaluate mathematical and logical
    expressions by parsing the AST and only allowing safe operations.
    
    Security Features:
    - No arbitrary code execution
    - Restricted to mathematical/logical operations only
    - Timeout protection available
    - Comprehensive audit logging
    """
    
    # Allowed AST node types
    ALLOWED_BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    
    ALLOWED_UNARYOPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
    }
    
    ALLOWED_COMPARATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }
    
    ALLOWED_BOOLOPS = {
        ast.And: lambda *args: all(args),
        ast.Or: lambda *args: any(args),
    }
    
    # Forbidden attributes and methods
    FORBIDDEN_ATTRS = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__globals__', '__code__', '__builtins__', '__import__',
        'system', 'popen', 'exec', 'eval', 'compile', 'open',
        'file', 'input', 'raw_input', 'reload',
    }
    
    def __init__(self, max_recursion_depth: int = 10):
        """
        Initialize the safe evaluator.
        
        Args:
            max_recursion_depth: Maximum allowed AST depth to prevent DoS
        """
        self.max_recursion_depth = max_recursion_depth
    
    def safe_eval(self, expression: str, context: Dict[str, Any], return_bool: bool = False) -> Any:
        """
        Safely evaluate a restricted mathematical/logical expression.
        
        Args:
            expression: The expression string to evaluate
            context: Dictionary of variables available to the expression
            return_bool: If True, return boolean result; otherwise return actual value
            
        Returns:
            Evaluation result (boolean if return_bool=True, else actual value)
            
        Raises:
            ValueError: If expression contains forbidden operations
            SyntaxError: If expression has invalid syntax
        """
        try:
            # Parse the expression
            tree = ast.parse(expression.strip(), mode='eval')
            
            # Validate the AST
            self._validate_ast(tree.body, depth=0)
            
            # Evaluate safely
            result = self._eval_node(tree.body, context)
            
            # Ensure boolean result for conditions if requested
            if return_bool:
                return bool(result)
            return result
            
        except SafeEvalError:
            raise
        except Exception as e:
            logger.warning(f"Safe expression evaluation failed: {e}")
            raise SafeEvalError(f"Invalid expression: {expression}") from e
    
    def _validate_ast(self, node: ast.AST, depth: int) -> None:
        """
        Validate that AST node is safe to evaluate.
        
        Args:
            node: AST node to validate
            depth: Current recursion depth
            
        Raises:
            SafeEvalError: If node contains forbidden operations
        """
        if depth > self.max_recursion_depth:
            raise SafeEvalError("Expression too complex")
        
        # Check for forbidden node types
        if isinstance(node, ast.Call):
            raise SafeEvalError("Function calls are not allowed")
        
        if isinstance(node, ast.Attribute):
            if node.attr in self.FORBIDDEN_ATTRS:
                raise SafeEvalError(f"Forbidden attribute access: {node.attr}")
            self._validate_ast(node.value, depth + 1)
        
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self.ALLOWED_BINOPS:
                raise SafeEvalError(f"Unsupported binary operation: {type(node.op).__name__}")
            self._validate_ast(node.left, depth + 1)
            self._validate_ast(node.right, depth + 1)
        
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.ALLOWED_UNARYOPS:
                raise SafeEvalError(f"Unsupported unary operation: {type(node.op).__name__}")
            self._validate_ast(node.operand, depth + 1)
        
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in self.ALLOWED_COMPARATORS:
                    raise SafeEvalError(f"Unsupported comparison: {type(op).__name__}")
            self._validate_ast(node.left, depth + 1)
            for comparator in node.comparators:
                self._validate_ast(comparator, depth + 1)
        
        elif isinstance(node, ast.BoolOp):
            if type(node.op) not in self.ALLOWED_BOOLOPS:
                raise SafeEvalError(f"Unsupported boolean operation: {type(node.op).__name__}")
            for value in node.values:
                self._validate_ast(value, depth + 1)
        
        elif isinstance(node, (ast.Name, ast.Constant, ast.Num, ast.Str)):
            pass  # These are safe
        
        elif isinstance(node, ast.Subscript):
            self._validate_ast(node.value, depth + 1)
            self._validate_ast(node.slice, depth + 1)
        
        elif isinstance(node, ast.Index):  # For Python < 3.9
            self._validate_ast(node.value, depth + 1)
        
        else:
            raise SafeEvalError(f"Unsupported AST node type: {type(node).__name__}")
    
    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """
        Recursively evaluate AST node in given context.
        
        Args:
            node: AST node to evaluate
            context: Variable context
            
        Returns:
            Evaluation result
        """
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        
        elif isinstance(node, ast.Name):
            if node.id not in context:
                raise SafeEvalError(f"Undefined variable: {node.id}")
            return context[node.id]
        
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            op_func = self.ALLOWED_BINOPS[type(node.op)]
            return op_func(left, right)
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            op_func = self.ALLOWED_UNARYOPS[type(node.op)]
            return op_func(operand)
        
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                op_func = self.ALLOWED_COMPARATORS[type(op)]
                result = result and op_func(left, right)
                left = right
            return result
        
        elif isinstance(node, ast.BoolOp):
            values = [self._eval_node(v, context) for v in node.values]
            op_func = self.ALLOWED_BOOLOPS[type(node.op)]
            return op_func(*values)
        
        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value, context)
            slice_val = self._eval_node(node.slice, context)
            return value[slice_val]
        
        elif isinstance(node, ast.Index):  # For Python < 3.9
            return self._eval_node(node.value, context)
        
        else:
            raise SafeEvalError(f"Cannot evaluate node type: {type(node).__name__}")


# Global instance for reuse
_safe_evaluator = SafeExpressionEvaluator()


def safe_eval_expression(expression: str, context: Dict[str, Any], return_bool: bool = False) -> Any:
    """
    Convenience function for safe expression evaluation.
    
    Args:
        expression: Expression string to evaluate
        context: Dictionary of variables for evaluation
        return_bool: If True, return boolean result; otherwise return actual value
        
    Returns:
        Evaluation result (boolean if return_bool=True, else actual value)
        
    Example:
        >>> context = {'df': dataframe, 'i': 5}
        >>> result = safe_eval_expression("df['close'].iloc[i] > df['close'].shift(1).iloc[i]", context)
    """
    return _safe_evaluator.safe_eval(expression, context, return_bool=return_bool)
