# Create users in a hard coded way
# Pre-setting user details: name, username, and password
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

names = ["Mor Tzabari", "Guy Mor"]
usernames = ["MorT", "GuyM"]
passwords = ["abc123", "def123"]

# Hash each password individually
hashed_passwords = stauth.Hasher(passwords).generate()

# Save hashed passwords to a file
file_path = Path(__file__).parent / "hashed_passwords.pkl"
with open(file_path, "wb") as file:
    pickle.dump(hashed_passwords, file)

print("Hashed passwords saved:", hashed_passwords)

