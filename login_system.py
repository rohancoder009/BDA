import sqlite3
import hashlib
import streamlit as st
from contextlib import contextmanager

DB_PATH = "users.db"  

def get_connection():
    """Connect to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None

@contextmanager
def get_db_cursor():
    conn = get_connection()
    if conn is None:
        yield None, None
        return

    cursor = conn.cursor()
    try:
        yield conn, cursor
    finally:
        cursor.close()
        conn.close()


def create_table():
    """Create users table if not exists"""
    with get_db_cursor() as (conn, cursor):
        if conn is None:
            return False
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            return True
        except sqlite3.Error as e:
            st.error(f"Error creating users table: {e}")
            return False


def hash_pwd(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def add_user(username, password):
    """Insert new user"""
    with get_db_cursor() as (conn, cursor):
        if conn is None:
            return False

        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hash_pwd(password))
            )
            conn.commit()
            return True

        except sqlite3.IntegrityError:
            return False  # username already exists

        except sqlite3.Error as e:
            st.error(f"Error adding user: {e}")
            return False


def check_user(username, password):
    """Validate login user"""
    with get_db_cursor() as (conn, cursor):
        if conn is None:
            return None

        try:
            cursor.execute(
                "SELECT id, username FROM users WHERE username = ? AND password = ?",
                (username, hash_pwd(password))
            )
            return cursor.fetchone()

        except sqlite3.Error as e:
            st.error(f"Error checking login: {e}")
            return None


def init_database():
    """Initialize DB"""
    return create_table()
