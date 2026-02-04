"""Tests for the main module."""

import pytest
from src.main import Calculator, calculate_sum, find_max, greet


class TestGreet:
    """Tests for the greet function."""

    def test_greet_basic(self):
        """Test basic greeting."""
        assert greet("World") == "Hello, World!"

    def test_greet_empty_name(self):
        """Test greeting with empty name."""
        assert greet("") == "Hello, !"

    def test_greet_special_characters(self):
        """Test greeting with special characters."""
        assert greet("John Doe") == "Hello, John Doe!"


class TestCalculateSum:
    """Tests for the calculate_sum function."""

    def test_sum_positive_numbers(self):
        """Test sum of positive numbers."""
        assert calculate_sum([1, 2, 3, 4, 5]) == 15

    def test_sum_empty_list(self):
        """Test sum of empty list."""
        assert calculate_sum([]) == 0

    def test_sum_negative_numbers(self):
        """Test sum with negative numbers."""
        assert calculate_sum([-1, -2, 3]) == 0

    def test_sum_single_element(self):
        """Test sum of single element."""
        assert calculate_sum([42]) == 42


class TestFindMax:
    """Tests for the find_max function."""

    def test_max_positive_numbers(self):
        """Test max of positive numbers."""
        assert find_max([1, 5, 3, 2, 4]) == 5

    def test_max_empty_list(self):
        """Test max of empty list."""
        assert find_max([]) is None

    def test_max_negative_numbers(self):
        """Test max with negative numbers."""
        assert find_max([-5, -1, -3]) == -1

    def test_max_single_element(self):
        """Test max of single element."""
        assert find_max([42]) == 42


class TestCalculator:
    """Tests for the Calculator class."""

    def test_initial_value(self):
        """Test calculator initialization."""
        calc = Calculator(10)
        assert calc.get_value() == 10

    def test_default_initial_value(self):
        """Test default initialization."""
        calc = Calculator()
        assert calc.get_value() == 0

    def test_add(self):
        """Test addition."""
        calc = Calculator(10)
        calc.add(5)
        assert calc.get_value() == 15

    def test_subtract(self):
        """Test subtraction."""
        calc = Calculator(10)
        calc.subtract(3)
        assert calc.get_value() == 7

    def test_multiply(self):
        """Test multiplication."""
        calc = Calculator(10)
        calc.multiply(3)
        assert calc.get_value() == 30

    def test_divide(self):
        """Test division."""
        calc = Calculator(10)
        calc.divide(2)
        assert calc.get_value() == 5

    def test_divide_by_zero(self):
        """Test division by zero raises error."""
        calc = Calculator(10)
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(0)

    def test_chaining(self):
        """Test method chaining."""
        calc = Calculator(10)
        result = calc.add(5).multiply(2).subtract(10).get_value()
        assert result == 20

    def test_reset(self):
        """Test reset functionality."""
        calc = Calculator(100)
        calc.reset()
        assert calc.get_value() == 0
