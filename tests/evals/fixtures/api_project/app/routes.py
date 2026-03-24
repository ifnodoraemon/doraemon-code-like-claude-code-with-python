"""API routes module."""

from typing import Any

from app.main import (
    create_item,
    create_user,
    delete_item,
    delete_user,
    get_item,
    get_user,
    list_items,
    list_users,
    update_item,
    update_user,
)


# Simulated HTTP response helpers
def json_response(data: Any, status_code: int = 200) -> dict[str, Any]:
    """Create a JSON response."""
    return {"status_code": status_code, "data": data}


def error_response(message: str, status_code: int = 400) -> dict[str, Any]:
    """Create an error response."""
    return {"status_code": status_code, "error": message}


# User routes
def route_create_user(username: str, email: str) -> dict[str, Any]:
    """POST /users - Create a new user."""
    if not username or not email:
        return error_response("Username and email are required", 400)

    user = create_user(username, email)
    return json_response(
        {"id": user.id, "username": user.username, "email": user.email},
        201,
    )


def route_get_user(user_id: int) -> dict[str, Any]:
    """GET /users/{user_id} - Get a user by ID."""
    user = get_user(user_id)
    if not user:
        return error_response("User not found", 404)

    return json_response({"id": user.id, "username": user.username, "email": user.email})


def route_list_users() -> dict[str, Any]:
    """GET /users - List all users."""
    users = list_users()
    return json_response([{"id": u.id, "username": u.username, "email": u.email} for u in users])


def route_update_user(user_id: int, **kwargs) -> dict[str, Any]:
    """PUT /users/{user_id} - Update a user."""
    user = update_user(user_id, **kwargs)
    if not user:
        return error_response("User not found", 404)

    return json_response({"id": user.id, "username": user.username, "email": user.email})


def route_delete_user(user_id: int) -> dict[str, Any]:
    """DELETE /users/{user_id} - Delete a user."""
    if delete_user(user_id):
        return json_response({"message": "User deleted"})
    return error_response("User not found", 404)


# Item routes
def route_create_item(
    name: str,
    description: str = None,
    price: float = 0.0,
    owner_id: int = None,
) -> dict[str, Any]:
    """POST /items - Create a new item."""
    if not name:
        return error_response("Name is required", 400)

    item = create_item(name, description, price, owner_id)
    return json_response(
        {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "price": item.price,
            "owner_id": item.owner_id,
        },
        201,
    )


def route_get_item(item_id: int) -> dict[str, Any]:
    """GET /items/{item_id} - Get an item by ID."""
    item = get_item(item_id)
    if not item:
        return error_response("Item not found", 404)

    return json_response(
        {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "price": item.price,
            "owner_id": item.owner_id,
        }
    )


def route_list_items(owner_id: int | None = None) -> dict[str, Any]:
    """GET /items - List items."""
    items = list_items(owner_id)
    return json_response(
        [
            {
                "id": i.id,
                "name": i.name,
                "description": i.description,
                "price": i.price,
                "owner_id": i.owner_id,
            }
            for i in items
        ]
    )


def route_update_item(item_id: int, **kwargs) -> dict[str, Any]:
    """PUT /items/{item_id} - Update an item."""
    item = update_item(item_id, **kwargs)
    if not item:
        return error_response("Item not found", 404)

    return json_response(
        {
            "id": item.id,
            "name": item.name,
            "description": item.description,
            "price": item.price,
            "owner_id": item.owner_id,
        }
    )


def route_delete_item(item_id: int) -> dict[str, Any]:
    """DELETE /items/{item_id} - Delete an item."""
    if delete_item(item_id):
        return json_response({"message": "Item deleted"})
    return error_response("Item not found", 404)
