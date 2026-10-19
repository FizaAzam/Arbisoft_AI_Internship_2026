import mysql.connector
import config

def get_connection():
    return mysql.connector.connect(
        host=config.HOST,
        user=config.USER,
        password=config.PASSWORD,
        database=config.DATABASE_NAME,
    )

def setup():
    connection = mysql.connector.connect(
        host=config.HOST,
        user=config.USER,
        password=config.PASSWORD,
    )
    try:
        cursor = connection.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DATABASE_NAME}")
        cursor.execute(f"USE {config.DATABASE_NAME}")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS menu_items(
            item_id INT AUTO_INCREMENT PRIMARY KEY,
            item_name VARCHAR(100) NOT NULL,
            PRICE DECIMAL(6, 2) NOT NULL
            )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders(
            order_id INT AUTO_INCREMENT PRIMARY KEY,
            customer_name VARCHAR(100) NOT NULL,
            order_date DATE NOT NULL
            )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_details(
            detail_id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT NOT NULL,
            item_id INT NOT NULL,
            quantity INT NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (item_id) REFERENCES menu_items(item_id)
            )
        """)
    finally:
        connection.close()
