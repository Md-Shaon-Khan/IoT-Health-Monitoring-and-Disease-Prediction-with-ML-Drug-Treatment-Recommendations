import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConfig:
    host = "localhost"
    user = "root"
    password = os.getenv("1234")  # Correct attribute name and assignment
    database = "health_db"

    pool_name = "health_pool"
    pool_size = 10
