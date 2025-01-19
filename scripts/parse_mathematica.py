import subprocess
from functools import reduce


def is_numeric(x: str) -> bool:

    if any(map(lambda s: s.isalpha(), x)):
        return False

    if x.isnumeric():
        return True

    try:
        num_q = eval(x)
        if isinstance(num_q, (int, float)):
            return True
    except Exception:
        return False

    return False


def parse_math_expression(expr: str) -> str:
    """
    Parses a human-written math expression string and ensures valid Python syntax by adding '*'
    only where it is necessary for mathematical operations, preserving contiguous variable names.

    Args:
        expr (str): A string representing the math expression.

    Returns:
        str: A Python-valid math expression.
    """
    # Ensure '(' ')'  are separate tokens
    tokens = create_tokens(expr)

    def needs_multiplication(prev: str, curr: str):
        """Determine if '*' is needed between two tokens."""
        op_symbols = ["+", "-", "/", "**", "="]
        if any([prev in op_symbols, curr in op_symbols]):
            return False
        if is_numeric(prev) and curr.isalpha():
            return True  # Between number and variable
        if prev.isalpha() and curr.isalpha():
            return True  # Between two variables
        if prev.endswith(")") and (is_numeric(curr) or curr.isalpha()):
            return True  # Between ')' and a variable/number
        if (is_numeric(prev) or prev.isalpha()) and curr.startswith("("):
            return True  # Between a variable/number and '('
        if prev.endswith(")") and curr.startswith("("):
            return True  # Between two parenthesized experssions ') * ('
        return False

    # Reconstruct expression with '*' where needed
    parsed_expr = []
    for i in range(len(tokens) - 1):
        parsed_expr.append(tokens[i])
        if needs_multiplication(tokens[i], tokens[i + 1]):
            parsed_expr.append("*")

    parsed_expr.append(tokens[-1])

    return "".join(parsed_expr)


def create_tokens(expr):
    m_tokens = ["(", ")", "+", "-", "/"]
    expr = reduce(lambda x, y: x.replace(y, f" {y} "), m_tokens, expr)
    # Replace '^' with '**' and split into tokens
    tokens = expr.replace("^", " ** ").replace("\n", "").split(" ")
    tokens = list(filter(lambda s: s != "", tokens))
    return tokens


def create_function(expr: str) -> str:

    tokens = create_tokens(expr)
    identifier = sorted(filter(lambda s: s.isidentifier(), set(tokens)))

    body = parse_math_expression(expr)

    template = f"def foo({', '.join(identifier)}):\n    val = {body}\n    return val"
    return template


# Example usage
input_expr = """1/12 ny (-3 xi yf zf - 3 xi yi zf + 3 xf yf zi + 3 xf yi zi) + 
 1/12 nx (-xf xi zf - xi^2 zf + 2 yf yi zf + 2 yi^2 zf + xf^2 zi + 
    xf xi zi - 2 yf^2 zi - 2 yf yi zi) + 
 1/12 nz (2 xi yf^2 - 2 xf yf yi + 2 xi yf yi - 2 xf yi^2 - xi zf^2 + 
    xf zf zi - xi zf zi + xf zi^2)"""

x_expression = """
-(1/6) nz (xf + xi) (-xi yf + xf yi) + 
1/6 ny (xf + xi) (-xi zf + xf zi) +
1/6 nx (xf (2 yf + yi) + xi (yf + 2 yi)) (zf - zi)
-(1/6) ny (yf + yi) (-yi zf + yf zi)
"""

y_centroid = """
1/6 nz (yf + yi) (xi yf - xf yi) 
- 1/6 nz (zf + zi) (xi zf - xf zi) - 
 1/6 nx (yf + yi) (-yi zf + yf zi) + 
 1/6 ny (xf - xi) (yf (2 zf + zi) + yi (zf + 2 zi))
"""

z_centroid = """
-(1/6) nx (xf + xi) (-xi yf + xf yi) - 
 1/6 ny (zf + zi) (xi zf - xf zi) + 
 1/6 nx (zf + zi) (yi zf - yf zi) + 
 1/6 nz (yf - yi) (xf (2 zf + zi) + xi (zf + 2 zi))
"""


area = """
1/2 nz (xi yf-xf yi)+1/2 ny (-xi zf+xf zi)+1/2 nx (yi zf-yf zi)
"""

parsed_expr = create_function(area)
print(parsed_expr)
subprocess.run(["black", "-c", parsed_expr], check=True)
