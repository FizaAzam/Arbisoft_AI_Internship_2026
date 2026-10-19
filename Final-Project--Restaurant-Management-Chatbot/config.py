import os
from dotenv import load_dotenv

load_dotenv()

HOST = "localhost"
USER = "root"
PASSWORD = os.getenv("PASSWORD")
DATABASE_NAME = "restaurant_db"
