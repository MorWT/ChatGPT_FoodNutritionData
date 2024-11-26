from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb+srv://morwainberg97:3TtM2fXksWKU6qll@cluster0.kesvk.mongodb.net/?retryWrites=true&w="
                     "majority&appName=Cluster0")
db = client["auth_demo"]  # Database name
users_collection = db["users"]  # Collection name


def init_db():
    """Initialize the database with indexes if needed."""
    users_collection.create_index("username", unique=True)
    users_collection.create_index("email", unique=True)


def get_user_by_username(username):
    """Retrieve a user document by username."""
    return users_collection.find_one({"username": username})


def add_user(username, email, hashed_password):
    """Add a new user to the database."""
    user = {"username": username, "email": email, "hashed_password": hashed_password}
    users_collection.insert_one(user)
