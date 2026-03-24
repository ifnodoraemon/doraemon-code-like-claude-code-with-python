import os

import pytest

from src.servers._services import document, office


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


def test_create_read_docx(temp_dir):
    path = os.path.join(temp_dir, "test.docx")
    content = "Hello World\nThis is a test."
    title = "My Document"

    # Create
    result = office.create_docx(path, content, title)
    assert "Successfully created" in result
    assert os.path.exists(path)

    # Read back (using existing document service)
    text = document.parse_docx(path)
    assert "My Document" in text
    assert "Hello World" in text
    assert "This is a test" in text


def test_create_read_xlsx(temp_dir):
    path = os.path.join(temp_dir, "test.xlsx")
    data = [["Name", "Age"], ["Alice", 30], ["Bob", 25]]

    # Create
    result = office.create_xlsx(path, data, "Employees")
    assert "Successfully created" in result
    assert os.path.exists(path)

    # Read back
    text = document.parse_xlsx(path)
    assert "Sheet: Employees" in text
    assert "Name\tAge" in text
    assert "Alice\t30" in text


def test_add_sheet_xlsx(temp_dir):
    path = os.path.join(temp_dir, "test_multi.xlsx")
    data1 = [["A", "B"], [1, 2]]
    data2 = [["X", "Y"], [9, 8]]

    office.create_xlsx(path, data1, "Sheet1")
    result = office.add_sheet_xlsx(path, data2, "Sheet2")
    assert "Successfully added sheet" in result

    text = document.parse_xlsx(path)
    assert "Sheet: Sheet1" in text
    assert "Sheet: Sheet2" in text
    assert "X\tY" in text


def test_create_read_pptx(temp_dir):
    path = os.path.join(temp_dir, "test.pptx")
    slides = [
        {"title": "Slide 1", "content": "Content 1"},
        {"title": "Slide 2", "content": "Point A\nPoint B"},
    ]

    # Create
    result = office.create_pptx(path, slides)
    assert "Successfully created" in result
    assert os.path.exists(path)

    # Read back
    text = document.parse_pptx(path)
    assert "Slide 1" in text
    assert "Content 1" in text
    assert "Point A" in text
