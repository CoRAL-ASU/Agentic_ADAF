import ast
import operator


class CalculationAgent:
    """Agent for performing numeric calculations."""

    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
    }

    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval_node(self, node):
        """Recursively evaluate a restricted AST node."""
        if isinstance(node, ast.BinOp):
            op = self._binary_ops.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operator")
            return op(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op = self._unary_ops.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported unary operator")
            return op(self._eval_node(node.operand))
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):  # for Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant")
        raise ValueError("Unsupported expression")

    def compute(self, expression):
        """Return the evaluated result of an expression."""
        try:
            tree = ast.parse(expression, mode="eval")
            return self._eval_node(tree.body)
        except Exception:
            return None

