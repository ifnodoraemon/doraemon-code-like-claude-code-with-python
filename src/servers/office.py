"""
Doraemon Office MCP Server

Provides tools for creating and editing Microsoft Office documents.
"""

import json
import logging
import os

from mcp.server.fastmcp import FastMCP

from src.core.logger import configure_root_logger
from src.core.security import validate_path
from src.services import office

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)

mcp = FastMCP("AgentOffice")


@mcp.tool()
def create_word_document(path: str, content: str, title: str = "") -> str:
    """
    Create a Microsoft Word (.docx) document.

    Args:
        path: Output file path (must end in .docx)
        content: The text content of the document. Paragraphs separated by newlines.
        title: Optional title for the document (Header 1)
    """
    try:
        valid_path = validate_path(path)
        if not valid_path.endswith(".docx"):
            return "Error: Path must end with .docx"

        # Ensure directory exists
        os.makedirs(os.path.dirname(valid_path), exist_ok=True)

        return office.create_docx(valid_path, content, title if title else None)
    except Exception as e:
        logger.error(f"Error creating Word doc: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def create_excel_sheet(path: str, data: str, sheet_name: str = "Sheet1") -> str:
    """
    Create a Microsoft Excel (.xlsx) spreadsheet.

    Args:
        path: Output file path (must end in .xlsx)
        data: JSON string representing a list of lists (rows).
              Example: '[["Name", "Age"], ["Alice", 30], ["Bob", 25]]'
        sheet_name: Name of the sheet
    """
    try:
        valid_path = validate_path(path)
        if not valid_path.endswith(".xlsx"):
            return "Error: Path must end with .xlsx"

        # Parse JSON data
        try:
            rows = json.loads(data)
            if not isinstance(rows, list):
                return "Error: Data must be a JSON list of lists."
        except json.JSONDecodeError:
            return "Error: Invalid JSON data."

        # Ensure directory exists
        os.makedirs(os.path.dirname(valid_path), exist_ok=True)

        return office.create_xlsx(valid_path, rows, sheet_name)
    except Exception as e:
        logger.error(f"Error creating Excel sheet: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def add_excel_sheet(path: str, data: str, sheet_name: str) -> str:
    """
    Add a new sheet to an existing Excel file.

    Args:
        path: Path to existing .xlsx file
        data: JSON string representing a list of lists (rows).
        sheet_name: Name of the new sheet
    """
    try:
        valid_path = validate_path(path)
        if not os.path.exists(valid_path):
            return "Error: File not found."

        # Parse JSON data
        try:
            rows = json.loads(data)
        except json.JSONDecodeError:
            return "Error: Invalid JSON data."

        return office.add_sheet_xlsx(valid_path, rows, sheet_name)
    except Exception as e:
        logger.error(f"Error adding Excel sheet: {e}")
        return f"Error: {str(e)}"


@mcp.tool()
def create_presentation(path: str, slides_json: str) -> str:
    """
    Create a PowerPoint (.pptx) presentation.

    Args:
        path: Output file path (must end in .pptx)
        slides_json: JSON string representing a list of slide objects.
                     Each object should have "title" and "content".
                     Example: '[{"title": "Intro", "content": "Hello World"}, {"title": "Slide 2", "content": "Bullet 1\\nBullet 2"}]'
    """
    try:
        valid_path = validate_path(path)
        if not valid_path.endswith(".pptx"):
            return "Error: Path must end with .pptx"

        # Parse JSON data
        try:
            slides = json.loads(slides_json)
            if not isinstance(slides, list):
                return "Error: Slides data must be a JSON list of objects."
        except json.JSONDecodeError:
            return "Error: Invalid JSON data."

        # Ensure directory exists
        os.makedirs(os.path.dirname(valid_path), exist_ok=True)

        return office.create_pptx(valid_path, slides)
    except Exception as e:
        logger.error(f"Error creating Presentation: {e}")
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
