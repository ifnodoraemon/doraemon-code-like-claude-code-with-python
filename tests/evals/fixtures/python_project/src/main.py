"""Main module for the test project."""

from typing import List, Optional


def greet(name: str) -> str:
    """
    Generate a greeting message.

    Args:
        name: The name to greet

    Returns:
        A greeting string
    """
    return f"Hello, {name}!"


def calculate_sum(numbers: List[int]) -> int:
    """
    Calculate the sum of a list of numbers.

    Args:
        numbers: List of integers to sum

    Returns:
        The sum of all numbers
    """
    return sum(numbers)


def find_max(numbers: List[int]) -> Optional[int]:
    """
    Find the maximum value in a list.

    Args:
        numbers: List of integers

    Returns:
        The maximum value, or None if list is empty
    """
    if not numbers:
        return None
    return max(numbers)


class Calculator:
    """A simple calculator class."""

    def __init__(self, initial_value: int = 0):
        """Initialize the calculator with an optional starting value."""
        self.value = initial_value

    def add(self, n: int) -> "Calculator":
        """Add a number to the current value."""
        self.value += n
        return self

    def subtract(self, n: int) -> "Calculator":
        """Subtract a number from the current value."""
        self.value -= n
        return self

    def multiply(self, n: int) -> "Calculator":
        """Multiply the current value by a number."""
        self.value *= n
        return self

    def divide(self, n: int) -> "Calculator":
        """Divide the current value by a number."""
        if n == 0:
            raise ValueError("Cannot divide by zero")
        self.value //= n
        return self

    def reset(self) -> "Calculator":
        """Reset the calculator to zero."""
        self.value = 0
        return self

    def get_value(self) -> int:
        """Get the current value."""
        return self.value


if __name__ == "__main__":
    print(greet("World"))
    print(f"Sum: {calculate_sum([1, 2, 3, 4, 5])}")
    print(f"Max: {find_max([1, 2, 3, 4, 5])}")

    calc = Calculator(10)
    result = calc.add(5).multiply(2).subtract(10).get_value()
    print(f"Calculator result: {result}")
