"""API application module."""

from dataclasses import dataclass


@dataclass
class User:
    """User model."""

    id: int
    username: str
    email: str
    is_active: bool = True


@dataclass
class Item:
    """Item model."""

    id: int
    name: str
    description: str | None = None
    price: float = 0.0
    owner_id: int | None = None


# In-memory storage
users_db: dict[int, User] = {}
items_db: dict[int, Item] = {}


def create_user(username: str, email: str) -> User:
    """Create a new user."""
    user_id = len(users_db) + 1
    user = User(id=user_id, username=username, email=email)
    users_db[user_id] = user
    return user


def get_user(user_id: int) -> User | None:
    """Get a user by ID."""
    return users_db.get(user_id)


def list_users() -> list[User]:
    """List all users."""
    return list(users_db.values())


def update_user(user_id: int, **kwargs) -> User | None:
    """Update a user."""
    user = users_db.get(user_id)
    if not user:
        return None

    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)

    return user


def delete_user(user_id: int) -> bool:
    """Delete a user."""
    if user_id in users_db:
        del users_db[user_id]
        return True
    return False


def create_item(
    name: str, description: str = None, price: float = 0.0, owner_id: int = None
) -> Item:
    """Create a new item."""
    item_id = len(items_db) + 1
    item = Item(
        id=item_id,
        name=name,
        description=description,
        price=price,
        owner_id=owner_id,
    )
    items_db[item_id] = item
    return item


def get_item(item_id: int) -> Item | None:
    """Get an item by ID."""
    return items_db.get(item_id)


def list_items(owner_id: int | None = None) -> list[Item]:
    """List items, optionally filtered by owner."""
    items = list(items_db.values())
    if owner_id is not None:
        items = [item for item in items if item.owner_id == owner_id]
    return items


def update_item(item_id: int, **kwargs) -> Item | None:
    """Update an item."""
    item = items_db.get(item_id)
    if not item:
        return None

    for key, value in kwargs.items():
        if hasattr(item, key):
            setattr(item, key, value)

    return item


def delete_item(item_id: int) -> bool:
    """Delete an item."""
    if item_id in items_db:
        del items_db[item_id]
        return True
    return False
