"""Compare factorial products without constructing enormous integer values.

The parser accepts a small mathematical grammar, never Python expressions. A
comparison either supplies an exact proof or leaves the answer unresolved.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from yugos.evaluation.grades import GradeCode

from ._outcome import Outcome

VERSION = "symbolic-products-v1"
MAX_TEXT = 65_536
MAX_TERMS = 4096
MAX_BITS = 131072
_TOKENS = re.compile(r"\\[A-Za-z]+|[0-9]+|[A-Za-z]|[^\s]")


def is_product_reference(reference):
    return "!" in reference and any(mark in reference for mark in (r"\prod", r"\cdots"))


class Unsupported(ValueError):
    """The bounded expression grammar cannot establish a comparison."""


def compact(text):
    for command in (r"\left", r"\right", r"\displaystyle", r"\,", r"\!"):
        text = text.replace(command, "")
    return re.sub(r"\s+", "", text)


def last_box(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += len(r"\boxed{")
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth > 64:
            return None
        if depth == 0:
            return text[start:end]
    return None


class Parser:
    def __init__(self, text):
        for command in (r"\left", r"\right", r"\displaystyle", r"\,", r"\!"):
            text = text.replace(command, "")
        self.tokens = _TOKENS.findall(text)
        self.pos = 0
        self.depth = 0
        if len(self.tokens) > 4096:
            raise Unsupported("expression too large")

    def take(self, token=None):
        if self.pos == len(self.tokens):
            raise Unsupported("incomplete expression")
        value = self.tokens[self.pos]
        if token is not None and value != token:
            raise Unsupported("unsupported expression")
        self.pos += 1
        return value

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def parse(self):
        result = self.expression()
        if self.peek() is not None:
            raise Unsupported("trailing expression")
        return result

    def expression(self):
        node = self.term()
        while self.peek() in {"+", "-"}:
            node = (self.take(), node, self.term())
        return node

    def term(self):
        node = self.power()
        while self.peek() in {r"\cdot", r"\times", "*", "/", r"\bmod"} or self.peek() in {
            "(",
            "{",
            r"\prod",
            r"\frac",
        }:
            op = self.take() if self.peek() not in {"(", "{", r"\prod", r"\frac"} else "*"
            node = ("mod" if op == r"\bmod" else "/" if op == "/" else "*", node, self.power())
        return node

    def power(self):
        node = self.atom()
        if self.peek() == "^":
            self.take()
            node = ("^", node, self.atom())
        if self.peek() == "!":
            self.take()
            node = ("!", node)
        return node

    def atom(self):
        self.depth += 1
        if self.depth > 64:
            raise Unsupported("expression too deep")
        try:
            token = self.take()
            if token in {"(", "{"}:
                node = self.expression()
                self.take(")" if token == "(" else "}")
                return node
            if token in {"-", "+"}:
                return (token, ("int", 0), self.power())
            if token == r"\frac":
                return ("/", self.atom(), self.atom())
            if token == r"\lfloor":
                node = self.expression()
                self.take(r"\rfloor")
                return ("floor", node)
            if token == r"\prod":
                self.take("_")
                self.take("{")
                variable = self.take()
                if not re.fullmatch("[A-Za-z]", variable):
                    raise Unsupported("unsupported product index")
                self.take("=")
                lower = self.expression()
                self.take("}")
                self.take("^")
                upper = self.atom()
                return ("prod", variable, lower, upper, self.power())
            if token.isdecimal() and len(token) <= 4096:
                return ("int", int(token))
            if re.fullmatch("[A-Za-z]", token):
                if self.peek() == "_":
                    self.take()
                    return ("indexed", token, self.atom())
                return ("var", token)
            raise Unsupported("unsupported operator")
        finally:
            self.depth -= 1


def number(node, environment):
    kind, *args = node
    if kind == "int":
        result = Fraction(args[0])
    elif kind == "var" and args[0] in environment:
        result = Fraction(environment[args[0]])
    elif kind == "floor":
        value = number(args[0], environment)
        result = Fraction(value.numerator // value.denominator)
    elif kind in {"+", "-", "*", "/", "^", "mod"}:
        a, b = (number(value, environment) for value in args)
        if kind == "+":
            result = a + b
        elif kind == "-":
            result = a - b
        elif kind == "*":
            result = a * b
        elif kind == "/":
            if not b:
                raise Unsupported("division by zero")
            result = a / b
        elif kind == "mod":
            if a.denominator != 1 or b.denominator != 1 or b <= 0:
                raise Unsupported("integer modulus required")
            result = a % b
        else:
            if (
                b.denominator != 1
                or abs(b) > MAX_BITS
                or (
                    max((abs(a.numerator) - 1).bit_length(), (a.denominator - 1).bit_length(), 1)
                    * abs(b)
                    + 1
                    > MAX_BITS
                )
            ):
                raise Unsupported("power exceeds comparison budget")
            result = a ** int(b)
    else:
        raise Unsupported("unresolved definition")
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > MAX_BITS:
        raise Unsupported("integer exceeds comparison budget")
    return result


def integer(node, environment):
    value = number(node, environment)
    if value.denominator != 1:
        raise Unsupported("integer required")
    return int(value)


@dataclass
class Product:
    coefficient: Fraction
    factorials: Counter

    def small_value(self, remaining):
        if max(remaining, default=0) > 64 or sum(remaining.values()) > 256:
            raise Unsupported("numeric comparison exceeds budget")
        value = self.coefficient
        for argument, count in remaining.items():
            for factor in range(2, argument + 1):
                value *= factor**count
                if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
                    raise Unsupported("numeric comparison exceeds budget")
        return value


def factors(node, environment, definitions, budget):
    budget[0] -= 1
    if budget[0] < 0:
        raise Unsupported("product exceeds comparison budget")
    kind, *args = node
    if kind == "!":
        arg = args[0]
        if arg[0] == "indexed" and arg[1] in definitions:
            if definitions[arg[1]] is None:
                raise Unsupported("unresolved definition")
            variable, expression = definitions[arg[1]]
            arg_environment = {**environment, variable: integer(arg[2], environment)}
            value = integer(expression, arg_environment)
        else:
            value = integer(arg, environment)
        if value < 0:
            raise Unsupported("negative factorial")
        return Product(Fraction(1), Counter({value: 1}) if value > 1 else Counter())
    if kind == "*":
        left, right = (factors(arg, environment, definitions, budget) for arg in args)
        return Product(left.coefficient * right.coefficient, left.factorials + right.factorials)
    if kind == "prod":
        variable, lower, upper, body = args
        lower, upper = integer(lower, environment), integer(upper, environment)
        if not 0 <= upper - lower < MAX_TERMS:
            raise Unsupported("unsupported product bounds")
        result = Product(Fraction(1), Counter())
        for index in range(lower, upper + 1):
            term = factors(body, {**environment, variable: index}, definitions, budget)
            result.coefficient *= term.coefficient
            result.factorials.update(term.factorials)
        return result
    return Product(number(node, environment), Counter())


def definitions(text):
    """Read explicit indexed equalities; ambiguous definitions remain unresolved."""
    found = {}
    exceptions = set(re.findall(r"([A-Za-z])_\{?[0-9]+\}?\s*=", text))
    for match in re.finditer(r"([A-Za-z])_(?:\{([A-Za-z])\}|([A-Za-z]))\s*=\s*([^,\n$]+)", text):
        name, braced, bare, raw = match.groups()
        variable = braced or bare
        if name in exceptions or re.search(r"\\(?:quad|qquad|ge|le|in|forall)\b|[<>]", raw):
            found[name] = None
            continue
        raw = re.split(r"\\(?:quad|qquad)|\\\]|\\\)", raw)[0]
        try:
            expression = Parser(raw).parse()
        except Unsupported:
            found[name] = None
            continue
        value = variable, expression
        if name in found and found[name] != value:
            raise Unsupported("ambiguous definition")
        found[name] = value
    return found


def parse_product(expression, context=""):
    normalized = compact(expression)
    # A geometric factorial sequence commits to both its first terms and its
    # final power. Other ellipses need an explicit bound rather than a guess.
    match = re.fullmatch(r"([0-9]+)!\\cdot([0-9]+)!\\cdots([0-9]+)\^\{([0-9]+)\}!", normalized)
    if match:
        first, second, base, end = map(int, match.groups())
        if first != base or second != base * base:
            raise Unsupported("ambiguous ellipsis")
        expression = rf"\prod_{{i=1}}^{{{end}}}({base}^i)!"
    environment = {}
    scalars = {}
    for match in re.finditer(r"(?<![A-Za-z0-9_\\])([A-Za-z])\s*=\s*([^,\n$]+)", context):
        name, raw = match.groups()
        raw = re.split(r"\\\]|\\\)", raw)[0]
        try:
            value = Parser(raw).parse()
        except Unsupported:
            value = None
        if name in scalars and scalars[name] != value:
            raise Unsupported("ambiguous scalar definition")
        scalars[name] = value
    # Resolve only explicit, acyclic numeric definitions within the same bounds.
    for _ in range(min(16, len(scalars))):
        for name, value in scalars.items():
            if value is None or name in environment:
                continue
            try:
                environment[name] = number(value, environment)
            except (Unsupported, ArithmeticError):
                pass
    return factors(Parser(expression).parse(), environment, definitions(context), [MAX_TERMS * 4])


def compare_products(reference, text, *, context=None):
    """Return None outside this grammar, otherwise an exact or unresolved grade."""
    if not is_product_reference(reference):
        return None
    details = {"grading_repair": VERSION, "reference_unchanged": True}
    expression = last_box(text[-MAX_TEXT:])
    try:
        gold = parse_product(reference)
    except (Unsupported, ArithmeticError, RecursionError):
        return Outcome(GradeCode.INVALID_GOLD, status="unavailable", details=details)
    try:
        if expression is None:
            raise Unsupported("missing final expression")
        literal = re.fullmatch(r"([+-]?)([0-9]+)", compact(expression))
        if literal and gold.coefficient > 0:
            digits = literal[2].lstrip("0")
            # n! >= 2^(n-1), whereas a d-digit integer is below 2^(4d).
            # Compare digit counts without converting a giant decimal literal.
            if (
                literal[1] == "-"
                or not digits
                or gold.coefficient >= 1
                and max(gold.factorials, default=1) - 1 >= 4 * len(digits)
            ):
                return Outcome(GradeCode.INCORRECT, details=details)
        answer = parse_product(expression, (text if context is None else context)[-MAX_TEXT:])
        gold_common = gold.factorials & answer.factorials
        left, right = gold.factorials - gold_common, answer.factorials - gold_common
        if gold.coefficient == answer.coefficient == 0:
            return Outcome(GradeCode.CORRECT, score=1, details=details)
        if gold.coefficient == answer.coefficient and left == right:
            return Outcome(GradeCode.CORRECT, score=1, details=details)
        if gold.coefficient > 0 and answer.coefficient <= 0:
            return Outcome(GradeCode.INCORRECT, details=details)
        if gold.coefficient == answer.coefficient and bool(left) != bool(right):
            return Outcome(GradeCode.INCORRECT, details=details)
        # n! * m! <= (n+m)!; strict for positive n,m. This proves single
        # factorial bounds while avoiding every factorial expansion.
        if gold.coefficient == answer.coefficient and left and right:
            a, b = sum(k * v for k, v in left.items()), sum(k * v for k, v in right.items())
            if (
                len(right) == 1
                and sum(right.values()) == 1
                and next(iter(right)) >= a
                or len(left) == 1
                and sum(left.values()) == 1
                and next(iter(left)) >= b
            ):
                return Outcome(GradeCode.INCORRECT, details=details)
            if sum(left.values()) == sum(right.values()):
                first, second = sorted(left.elements()), sorted(right.elements())
                if all(a <= b for a, b in zip(first, second)) or all(
                    a >= b for a, b in zip(first, second)
                ):
                    return Outcome(GradeCode.INCORRECT, details=details)
        if not answer.factorials and gold.coefficient > 0 and answer.coefficient > 0:
            largest = max(gold.factorials, default=1)
            threshold = (answer.coefficient / gold.coefficient).numerator.bit_length()
            if largest - 1 >= threshold:
                return Outcome(GradeCode.INCORRECT, details=details)
        matched = gold.small_value(left) == answer.small_value(right)
        return Outcome(
            GradeCode.CORRECT if matched else GradeCode.INCORRECT,
            score=float(matched),
            details=details,
        )
    except (Unsupported, ArithmeticError, RecursionError):
        pass
    return Outcome(
        GradeCode.NEEDS_REVIEW,
        status="unavailable",
        details={**details, "cause": "symbolic_comparison_unresolved"},
    )
