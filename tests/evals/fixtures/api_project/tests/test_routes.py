"""Tests for API routes."""

import pytest
from app.main import users_db, items_db
from app.routes import (
    route_create_user,
    route_get_user,
    route_list_users,
    route_update_user,
    route_delete_user,
    route_create_item,
    route_get_item,
    route_list_items,
    route_update_item,
    route_delete_item,
)


@pytest.fixture(autouse=True)
def clear_databases():
    """Clear databases before each test."""
    users_db.clear()
    items_db.clear()
    yield
    users_db.clear()
    items_db.clear()


class TestUserRoutes:
    """Tests for user routes."""

    def test_create_user(self):
        """Test creating a user."""
        response = route_create_user("testuser", "test@example.com")
        assert response["status_code"] == 201
        assert response["data"]["username"] == "testuser"
        assert response["data"]["email"] == "test@example.com"

    def test_create_user_missing_fields(self):
        """Test creating user with missing fields."""
        response = route_create_user("", "test@example.com")
        assert response["status_code"] == 400

    def test_get_user(self):
        """Test getting a user."""
        route_create_user("testuser", "test@example.com")
        response = route_get_user(1)
        assert response["status_code"] == 200
        assert response["data"]["username"] == "testuser"

    def test_get_user_not_found(self):
        """Test getting non-existent user."""
        response = route_get_user(999)
        assert response["status_code"] == 404

    def test_list_users(self):
        """Test listing users."""
        route_create_user("user1", "user1@example.com")
        route_create_user("user2", "user2@example.com")
        response = route_list_users()
        assert response["status_code"] == 200
        assert len(response["data"]) == 2

    def test_update_user(self):
        """Test updating a user."""
        route_create_user("testuser", "test@example.com")
        response = route_update_user(1, username="updated")
        assert response["status_code"] == 200
        assert response["data"]["username"] == "updated"

    def test_delete_user(self):
        """Test deleting a user."""
        route_create_user("testuser", "test@example.com")
        response = route_delete_user(1)
        assert response["status_code"] == 200

        # Verify deletion
        response = route_get_user(1)
        assert response["status_code"] == 404


class TestItemRoutes:
    """Tests for item routes."""

    def test_create_item(self):
        """Test creating an item."""
        response = route_create_item("Test Item", "A test item", 9.99)
        assert response["status_code"] == 201
        assert response["data"]["name"] == "Test Item"
        assert response["data"]["price"] == 9.99

    def test_create_item_missing_name(self):
        """Test creating item without name."""
        response = route_create_item("")
        assert response["status_code"] == 400

    def test_get_item(self):
        """Test getting an item."""
        route_create_item("Test Item", "A test item", 9.99)
        response = route_get_item(1)
        assert response["status_code"] == 200
        assert response["data"]["name"] == "Test Item"

    def test_list_items(self):
        """Test listing items."""
        route_create_item("Item 1", price=10.0)
        route_create_item("Item 2", price=20.0)
        response = route_list_items()
        assert response["status_code"] == 200
        assert len(response["data"]) == 2

    def test_list_items_by_owner(self):
        """Test listing items by owner."""
        route_create_user("owner", "owner@example.com")
        route_create_item("Item 1", owner_id=1)
        route_create_item("Item 2", owner_id=1)
        route_create_item("Item 3", owner_id=None)

        response = route_list_items(owner_id=1)
        assert response["status_code"] == 200
        assert len(response["data"]) == 2

    def test_update_item(self):
        """Test updating an item."""
        route_create_item("Test Item", price=10.0)
        response = route_update_item(1, price=15.0)
        assert response["status_code"] == 200
        assert response["data"]["price"] == 15.0

    def test_delete_item(self):
        """Test deleting an item."""
        route_create_item("Test Item")
        response = route_delete_item(1)
        assert response["status_code"] == 200

        # Verify deletion
        response = route_get_item(1)
        assert response["status_code"] == 404
