import os
import time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from app.database import Base, engine, init_db

def wait_for_db():
    """Ждём пока PostgreSQL запустится"""
    attempts = 0
    while attempts < 30:
        try:
            # Пробуем подключиться
            with engine.connect() as conn:
                # Проверяем что можем выполнить запрос
                conn.execute(text("SELECT 1"))
            print("Database is ready!")
            return True
        except OperationalError as e:
            attempts += 1
            print(f"Database not ready, waiting... (attempt {attempts}/30)")
            print(f"Error: {e}")
            time.sleep(2)
    return False

if __name__ == "__main__":
    print(f"Connecting to database: {os.getenv('DATABASE_URL', 'Not set')}")
    
    if wait_for_db():
        try:
            init_db()
            print("Database initialized successfully!")
        except Exception as e:
            print(f"Error initializing database: {e}")
            exit(1)
    else:
        print("Failed to connect to database after 30 attempts")
        exit(1)