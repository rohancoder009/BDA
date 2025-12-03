import mysql.connector
import hashlib
import streamlit as st
from contextlib import contextmanager
from dotenv import load_dotenv
import os
DB_HOST = st.secrets["DB_HOST"]
DB_USER = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
DB_NAME = st.secrets["DB_NAME"]
DB_Port= st.secrets["db_port"]

def get_connection():
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port = int(DB_Port)
        )
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        return None
@contextmanager
def get_db_cursor():
    """Context manager for database operations"""
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
    """Create users table if it doesn't exist"""
    with get_db_cursor() as (conn, cursor):
        if conn is None:
            return False
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            return True
        except mysql.connector.Error as e:
            st.error(f"Error creating table: {e}")
            return False

def hash_pwd(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    """Add new user to database"""
    with get_db_cursor() as (conn, cursor):
        if conn is None:
            return False
        
        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, hash_pwd(password))
            )
            conn.commit()
            return True
        except mysql.connector.IntegrityError:
            return False  # Username already exists
        except mysql.connector.Error as e:
            st.error(f"Error adding user: {e}")
            return False

def check_user(username, password):
    """Verify user credentials"""
    with get_db_cursor() as (conn, cursor):
        if conn is None:
            return None
        
        try:
            cursor.execute(
                "SELECT id, username FROM users WHERE username = %s AND password = %s",
                (username, hash_pwd(password))
            )
            user = cursor.fetchone()
            return user
        except mysql.connector.Error as e:
            st.error(f"Error checking user: {e}")
            return None

def init_database():
    """Initialize database and create tables"""
    return create_table()
