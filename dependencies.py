import streamlit as st
import bcrypt
from create_DB import get_user_by_username, add_user


def sign_up():
    """Sign up a new user."""
    st.subheader("Sign Up")

    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter your Username")
        email = st.text_input("Email", placeholder="Enter your Email")
        password1 = st.text_input("Password", placeholder="Enter your Password", type="password")
        password2 = st.text_input("Confirm Password", placeholder="Confirm your Password", type="password")
        submit_button = st.form_submit_button("Sign Up")

    if submit_button:
        if password1 != password2:
            st.error("Passwords do not match!")
            return

        if get_user_by_username(username):
            st.error("Username already exists!")
            return

        # Hash the password
        hashed_password = bcrypt.hashpw(password1.encode('utf-8'), bcrypt.gensalt())

        # Add user to the database
        add_user(username, email, hashed_password)
        st.success("You have successfully signed up! Please log in.")


def login():
    """Log in an existing user."""
    st.subheader("Login")

    username = st.text_input("Username", placeholder="Enter your Username")
    password = st.text_input("Password", placeholder="Enter your Password", type="password")

    if st.button("Login"):
        # Fetch user from the database
        user = get_user_by_username(username)
        if not user:
            st.error("User does not exist!")
            return

        hashed_password = user["hashed_password"]

        # Validate password
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
            # Update session state
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["page"] = "main"
            st.success(f"Welcome, {username}!")
            st.write(f"DEBUG: logged_in = {st.session_state['logged_in']}, page = {st.session_state['page']}")
        else:
            st.error("Incorrect password!")


def logout():
    """Log out the current user."""
    st.session_state["logged_in"] = False
    st.session_state["page"] = "login"
    st.session_state["username"] = ""
