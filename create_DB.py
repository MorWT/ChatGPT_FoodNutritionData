import sqlite3
import streamlit as st


# Initialize the database
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


# Add a new user
def add_user(email, username, hashed_password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (email, username, hashed_password) VALUES (?, ?, ?)",
                   (email, username, hashed_password))
    conn.commit()
    conn.close()


def get_user_password(username):
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]  # Return the hashed password
        return None  # User not found
    except Exception as e:
        st.write(f"DEBUG: Database error = {e}")
        return None


