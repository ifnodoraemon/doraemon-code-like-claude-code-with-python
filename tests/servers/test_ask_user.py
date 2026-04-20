"""Tests for src/servers/ask_user.py"""

from unittest.mock import patch

import pytest

from src.servers.ask_user import ask_user


class TestAskUserFreeText:
    @patch("src.servers.ask_user.Prompt.ask", return_value="my answer")
    def test_free_text(self, mock_ask):
        result = ask_user("What is your name?")
        assert result == "my answer"
        mock_ask.assert_called_once()

    @patch("src.servers.ask_user.Prompt.ask", return_value="  trimmed  ")
    def test_free_text_trims_whitespace(self, mock_ask):
        result = ask_user("Question?")
        assert result == "trimmed"


class TestAskUserWithOptions:
    @patch("src.servers.ask_user.Prompt.ask", return_value="1")
    def test_select_first_option(self, mock_ask):
        result = ask_user("Choose:", options="Yes,No,Maybe")
        assert result == "Yes"

    @patch("src.servers.ask_user.Prompt.ask", return_value="2")
    def test_select_second_option(self, mock_ask):
        result = ask_user("Choose:", options="A,B,C")
        assert result == "B"

    @patch("src.servers.ask_user.Prompt.ask", return_value="3")
    def test_select_third_option(self, mock_ask):
        result = ask_user("Choose:", options="A,B,C")
        assert result == "C"

    @patch("src.servers.ask_user.Prompt.ask", side_effect=["3", "custom input"])
    def test_select_other_custom_input(self, mock_ask):
        result = ask_user("Choose:", options="Yes,No")
        assert result == "custom input"

    @patch("src.servers.ask_user.Prompt.ask", return_value="direct text")
    def test_non_numeric_treated_as_text(self, mock_ask):
        result = ask_user("Choose:", options="Yes,No")
        assert result == "direct text"

    @patch("src.servers.ask_user.Prompt.ask", return_value="0")
    def test_out_of_range_returns_raw(self, mock_ask):
        result = ask_user("Choose:", options="Yes,No")
        assert result == "0"


class TestAskUserMultiSelect:
    @patch("src.servers.ask_user.Prompt.ask", return_value="1,2")
    def test_multi_select_two(self, mock_ask):
        result = ask_user("Choose:", options="A,B,C", multi_select=True)
        assert "A" in result
        assert "B" in result

    @patch("src.servers.ask_user.Prompt.ask", side_effect=["1,3", "custom"])
    def test_multi_select_with_custom(self, mock_ask):
        result = ask_user("Choose:", options="A,B", multi_select=True)
        assert "A" in result

    @patch("src.servers.ask_user.Prompt.ask", return_value="yes_text")
    def test_multi_select_non_numeric_as_literal(self, mock_ask):
        result = ask_user("Choose:", options="A,B", multi_select=True)
        assert "yes_text" in result


class TestAskUserEmptyOptions:
    @patch("src.servers.ask_user.Prompt.ask", return_value="free text")
    def test_empty_options_string(self, mock_ask):
        result = ask_user("Q?", options="")
        assert result == "free text"

    @patch("src.servers.ask_user.Prompt.ask", return_value="free text")
    def test_whitespace_only_options(self, mock_ask):
        result = ask_user("Q?", options="   ")
        assert result == "free text"

    @patch("src.servers.ask_user.Prompt.ask", return_value="free text")
    def test_commas_only_options(self, mock_ask):
        result = ask_user("Q?", options=",,,")
        assert result == "free text"
