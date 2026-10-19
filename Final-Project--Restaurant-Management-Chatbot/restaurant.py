from datetime import datetime
from langchain_core.tools import tool

from db import get_connection

@tool
def view_menu() -> str:
    """Show every dish on the restaurant menu with its ID and price."""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT item_id, item_name, price FROM menu_items")
        rows = cursor.fetchall()  #get all the rows that MySQL returned when SELECT was called

        if not rows:
            return "The menu is empty. No items have been added yet."

        lines = []
        for row in rows:
            item_id, item_name, price = row
            lines.append(f"#{item_id} | {item_name} | Rs. {price}")
        return "\n".join(lines)
    finally:
        connection.close()

@tool
def add_menu_item(name: str, price: float) -> str:
    """Add a new dish to the restaurant menu.

    Args:
        name: The dish name, for example "Biryani".
        price: The price in rupees, for example 450.0.
    """
    if price < 0:
        return "Error: price cannot be negative."

    connection = get_connection()
    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT item_name FROM menu_items WHERE item_name = %s LIMIT 1",
            (name,)
        )
        if cursor.fetchone() is not None:
            return f"'{name}' is already on the menu. Nothing was added."

        cursor.execute(
            "INSERT INTO menu_items(item_name, price) VALUES (%s, %s)",
            (name, price)
        )
        connection.commit()
        return f"Menu item '{name}' added successfully at Rs. {price}."
    finally:
        connection.close()

@tool
def place_order(customer_name: str, items: list[dict]) -> str:
    """Place a new order for a customer.

    Args:
        customer_name: The name of the customer, for example "Ali".
        items: The dishes being ordered, as a list of dictionaries. Each one
            needs an "item_name" that matches a dish already on the menu and a
            whole number "quantity", for example
            [{"item_name": "Biryani", "quantity": 2}].
    """
    if not items:
        return "Error: an order must contain at least one item."

    connection = get_connection()
    try:
        cursor = connection.cursor()

        # Check every item first, so a bad name never leaves a half-made order behind.
        resolved = []
        for item in items:
            item_name = item.get("item_name")
            quantity = item.get("quantity")

            if not item_name:
                return "Error: every item needs an 'item_name'."
            if not isinstance(quantity, int):
                return f"Error: quantity for '{item_name}' must be a whole number."
            if quantity <= 0:
                return f"Error: quantity for '{item_name}' must be greater than 0."


            cursor.execute(
                "SELECT item_id FROM menu_items WHERE item_name = %s LIMIT 1",
                (item_name,)
            )
            match = cursor.fetchone()  
            if match is None:
                return (
                    f"Error: '{item_name}' is not on the menu. "
                    f"The menu currently has:\n{view_menu.invoke({})}"
                )

            resolved.append((match[0], item_name, quantity))

        order_date = datetime.now().date()
        cursor.execute(
            "INSERT INTO orders(customer_name, order_date) VALUES (%s, %s)",
            (customer_name, order_date)
        )
        order_id = cursor.lastrowid  

        for item_id, item_name, quantity in resolved:
            cursor.execute(
                "INSERT INTO order_details(order_id, item_id, quantity) VALUES(%s,%s,%s)",
                (order_id, item_id, quantity)
            )

        connection.commit()

        summary = ", ".join(f"{name} x{qty}" for _, name, qty in resolved)
        return f"Order #{order_id} placed for {customer_name} on {order_date}: {summary}."
    finally:
        connection.close()

@tool
def view_orders() -> str:
    """Show every order that has been placed, with its items and quantities."""
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT orders.order_id, orders.customer_name, orders.order_date, menu_items.item_name, order_details.quantity
            FROM orders
            JOIN order_details ON orders.order_id = order_details.order_id
            JOIN menu_items ON order_details.item_id = menu_items.item_id
            ORDER BY orders.order_id
        """)
        rows = cursor.fetchall()

        if not rows:
            return "No orders have been placed yet."

        lines = []
        for row in rows:
            order_id, customer_name, order_date, item_name, quantity = row
            lines.append(f"Order #{order_id} | {customer_name} | {order_date} | {item_name} x{quantity}")
        return "\n".join(lines)
    finally:
        connection.close()


MENU_TOOLS = [view_menu, add_menu_item]
ORDER_TOOLS = [place_order, view_orders, view_menu]