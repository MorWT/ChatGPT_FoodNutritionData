import bcrypt
import streamlit as st
from create_DB import add_user, get_user_password
from utils import validate_email


# Sign-up function
def sign_up():
    st.subheader("Sign Up")

    with st.form("signup_form", clear_on_submit=True):
        email = st.text_input("Email", placeholder="Enter your Email")
        username = st.text_input("Username", placeholder="Enter your Username")
        password1 = st.text_input("Password", placeholder="Enter your Password", type="password")
        password2 = st.text_input("Confirm Password", placeholder="Confirm your Password", type="password")
        submit_button = st.form_submit_button("Sign Up")

    if submit_button:
        if password1 != password2:
            st.error("Passwords do not match!")
            return

        if not validate_email(email):
            st.error("Invalid email format!")
            return

        hashed_password = bcrypt.hashpw(password1.encode('utf-8'), bcrypt.gensalt())

        try:
            add_user(email, username, hashed_password)
            st.success("You have successfully signed up! Please log in.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


# Login function
# def login():
#     st.subheader("Login")
#
#     username = st.text_input("Username", placeholder="Enter your Username")
#     password = st.text_input("Password", placeholder="Enter your Password", type="password")
#
#     if st.button("Login"):
#         st.write(f"DEBUG: Username entered = {username}, Password entered = {password}")
#         hashed_password = get_user_password(username)
#         st.write(f"DEBUG: Hashed Password retrieved = {hashed_password}")
#
#         if hashed_password:  # Check if hashed password exists in the DB
#             # Check password validation
#             is_valid = bcrypt.checkpw(password.encode('utf-8'), hashed_password)
#             st.write(f"DEBUG: Password validation result = {is_valid}")
#
#             if is_valid:
#                 st.session_state["logged_in"] = True
#                 st.session_state["username"] = username
#                 st.session_state["page"] = "main"
#                 st.success(f"Welcome, {username}!")
#             else:
#                 st.error("Incorrect password!")
#         else:
#             st.error("User does not exist!")


def login():
    st.subheader("Login")

    username = st.text_input("Username", placeholder="Enter your Username")
    password = st.text_input("Password", placeholder="Enter your Password", type="password")

    if st.button("Login"):
        st.write(f"DEBUG: Username entered = {username}, Password entered = {password}")

        # Fetch hashed password from the database
        hashed_password = get_user_password(username)
        st.write(f"DEBUG: Hashed Password retrieved = {hashed_password}")

        if hashed_password:  # If the user exists
            is_valid = bcrypt.checkpw(password.encode('utf-8'), hashed_password)
            st.write(f"DEBUG: Password validation result = {is_valid}")
            if is_valid:
                # Successful login
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["page"] = "main"
                st.success(f"Welcome, {username}!")
            else:
                st.error("Incorrect password!")
        else:
            st.error("User does not exist!")




