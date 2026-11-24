"""Store and retrieve simulated orders using SQLite."""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORDERS_PATH = PROJECT_ROOT / "store" / "orders.json"

# Override this path when using a persistent Docker volume.
DB_PATH = Path(
    os.environ.get(
        "SUPPORT_DB_PATH",
        str(PROJECT_ROOT / "data" / "support.sqlite"),
    )
).expanduser()


@contextmanager
def connect_db():
    """Open a connection, commit successful writes, and always close."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    """Create the orders table if it does not already exist."""
    with connect_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                order_json TEXT NOT NULL
            )
            """
        )


def seed_orders():
    """Insert or update source orders without deleting other records."""
    source_orders = json.loads(ORDERS_PATH.read_text(encoding="utf-8"))

    if not isinstance(source_orders, list):
        raise ValueError("orders.json must contain a list.")

    order_ids = [order["order_id"] for order in source_orders]
    if len(order_ids) != len(set(order_ids)):
        raise ValueError("Duplicate order IDs found in orders.json.")

    rows = [
        (order["order_id"], json.dumps(order, ensure_ascii=False))
        for order in source_orders
    ]

    initialize_database()

    with connect_db() as connection:
        connection.executemany(
            """
            INSERT INTO orders (order_id, order_json)
            VALUES (?, ?)
            ON CONFLICT(order_id)
            DO UPDATE SET order_json = excluded.order_json
            """,
            rows,
        )

    return len(rows)


def get_order(order_id: str) -> dict | None:
    """Return one order, or None if the order ID does not exist."""
    with connect_db() as connection:
        row = connection.execute(
            "SELECT order_json FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()

    return json.loads(row["order_json"]) if row else None


def list_orders() -> list[dict]:
    """Return all orders sorted by order ID."""
    with connect_db() as connection:
        rows = connection.execute(
            "SELECT order_json FROM orders ORDER BY order_id"
        ).fetchall()

    return [json.loads(row["order_json"]) for row in rows]


if __name__ == "__main__":
    from store.rules import check_request

    count = seed_orders()
    print(f"Database: {DB_PATH}")
    print(f"Source orders imported: {count}")

    # Verify that storage preserves every source field and value.
    source_orders = json.loads(ORDERS_PATH.read_text(encoding="utf-8"))
    for source_order in source_orders:
        assert get_order(source_order["order_id"]) == source_order
    print("All source orders match database records.")

    # Repeated seeding must not create duplicate records.
    count_before = len(list_orders())
    seed_orders()
    assert len(list_orders()) == count_before
    print("Repeated seeding: no duplicate orders.")

    assert get_order("ORDER-DOES-NOT-EXIST") is None
    print("Missing order returns: None")

    # Verify compatibility with Member C's rule engine.
    order = get_order("ORD-1001")
    result = check_request("cancel_order", order)
    assert result.decision == "eligible"
    print(f"ORD-1001 cancellation decision: {result.decision}")
    print("Database checks passed.")