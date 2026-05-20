"""Predefined arithmetic tools exposed to the LangGraph agent."""

from __future__ import annotations

from langchain_core.tools import tool

_EPS = 1e-15


@tool
def plus(a: float, b: float) -> float:
    """Add two numbers. Use for addition questions (e.g. 'what is 5 plus 3')."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first. Use for subtraction questions."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Use for multiplication questions."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """
    Divide the first number by the second.
    Returns a string result or an error message if dividing by zero.
    """
    if abs(b) < _EPS:
        return "Error: division by zero is undefined."
    result = a / b
    return str(result)


def get_math_tools() -> list:
    """Return all math tools for binding to the chat model."""
    return [plus, subtract, multiply, divide]
