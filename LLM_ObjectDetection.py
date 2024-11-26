import os
import base64
import requests
import pandas as pd
import streamlit as st
from urllib.parse import urlparse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image
import json
from datetime import datetime
from dependencies import sign_up, logout
from create_DB import get_user_by_username
import bcrypt


OPENAI_API_KEY = ('sk-proj-zRcdPL6FzeYco9zbuJA7IAGLRGUKpLDwA8mWE170y1tQbqdAxTAQnIy69LIN4chYjyTUMGzWA9T3BlbkFJySLP3EvBqq'
                  'PalucM9_nLTD6m_q_465dK9ArhMDcr2Z1DCaD5KrgrRbTkdE1_RK8H2rJUFQKeEA')

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "page" not in st.session_state:
    st.session_state["page"] = "login"
if "username" not in st.session_state:
    st.session_state["username"] = ""


# Main application logic
def main_app():
    st.sidebar.button("Logout", on_click=logout)
    st.sidebar.write(f"Welcome, {st.session_state['username']}")
    st.title("Food Nutrition Assistant")

    class NutritionData(BaseModel):
        name: str = Field(description="The name of a food item")
        calories_per_100g: float = Field(description="The calories value of the food item per 100g")
        fat_per_100g: float = Field(description="The fat value of the food item per 100g")
        protein_per_100g: float = Field(description="The protein value of the food item per 100g")
        weight_grams: float = Field(description="The estimated weight of the food item in grams")

    class ImageRecognitionResult(BaseModel):
        food_items: list[NutritionData]

    def is_url(path):
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def download_image(image_path):
        response = requests.get(image_path, stream=True)
        if response.status_code == 200:
            local_filename = "downloaded_image.jpg"
            with open(local_filename, "wb") as img_file:
                for chunk in response.iter_content(chunk_size=8192):
                    img_file.write(chunk)
            return local_filename
        else:
            raise Exception(f"Failed to download image from URL: {image_path}")

    def resize_image_to_max_500px(image_path):
        with Image.open(image_path) as img:
            max_side = max(img.size)
            if max_side <= 500:
                return image_path
            ratio = 500 / max_side
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            resized_image_path = "resized_" + os.path.basename(image_path)
            resized_img.save(resized_image_path)
            return resized_image_path

    def process_image(image_path):
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

        if is_url(image_path):
            image_path = download_image(image_path)

        image_path = resize_image_to_max_500px(image_path)

        # Step 1: Read the image in binary mode and encode it in Base64
        with open(image_path, "rb") as img_file:
            image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Step 2: Update the prompt to include the Base64-encoded image
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the weather in this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert nutrition assistant that specializes in identifying food items from images. "
                       "Given an image of food, you will analyze it to identify all the types of food and ingredients. "
                       "Once identified, you will provide nutritional details such as name, calories, fat, and protein "
                       "per 100g. Base64-encoded images will be provided"),
            human_message
        ])

        # Step 3: Create a chain and pass the Base64 image
        chain = prompt | llm.with_structured_output(ImageRecognitionResult)
        result = chain.invoke({
            "image_b64": image_b64  # Pass the Base64-encoded image string here
        })

        return result.model_dump_json(indent=2)

    def display_nutrition_values(image_path, df):
        try:
            st.image(image_path, caption="Uploaded Image", use_container_width=True)
            st.write("Processing the image...")
            result = process_image(image_path)
            result_dict = json.loads(result)
            result_data = ImageRecognitionResult(**result_dict)
            meal_data = []
            total_calories = 0
            for item in result_data.food_items:
                item_calories = (item.calories_per_100g / 100) * item.weight_grams
                total_calories += item_calories
                meal_data.append({
                    "Upload Date": datetime.now().strftime("%d.%m.%y"),
                    "Upload Hour": datetime.now().strftime("%H:%M"),
                    "Food Name": item.name.capitalize(),
                    "Calories (per 100g)": item.calories_per_100g,
                    "Fat (per 100g)": item.fat_per_100g,
                    "Protein (per 100g)": item.protein_per_100g,
                    "Weight (grams)": item.weight_grams,
                    "Calories (in image)": round(item_calories, 2),
                })
            df = pd.concat([df, pd.DataFrame(meal_data)], ignore_index=True)
            st.markdown("### Meal Nutrition Details")
            st.dataframe(pd.DataFrame(meal_data))
            st.markdown(f"### Total Calories for this Meal: {round(total_calories, 2)}")
            return df
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return df

    def calculate_daily_calories(gender, weight, height, age, activity_level):
        if gender == "Male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        activity_multiplier = {
            "Sedentary": 1.2,
            "Lightly active": 1.375,
            "Moderately active": 1.55,
            "Very active": 1.725,
            "Extra active": 1.9
        }
        return bmr * activity_multiplier[activity_level]

    # Sidebar for user details
    st.sidebar.markdown("## Enter Your Details")
    user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    user_weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    user_height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    user_age = st.sidebar.number_input("Age", min_value=10, max_value=120, value=30)
    user_activity_level = st.sidebar.selectbox(
        "Activity Level",
        ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"]
    )

    daily_calories = calculate_daily_calories(user_gender, user_weight, user_height, user_age, user_activity_level)

    # Initialize session state
    if "nutrition_df" not in st.session_state:
        df_columns = [
            "Food Name", "Calories (per 100g)", "Fat (per 100g)", "Protein (per 100g)",
            "Weight (grams)", "Calories (in image)", "Upload Date", "Upload Hour"
        ]
        st.session_state.nutrition_df = pd.DataFrame(columns=df_columns)

    if "processed_images" not in st.session_state:
        st.session_state.processed_images = set()

    # Input for the first image
    image_path = st.text_input("Provide a link to an image:")
    if st.button("Process Image"):
        if image_path not in st.session_state.processed_images:
            st.session_state.nutrition_df = display_nutrition_values(image_path, st.session_state.nutrition_df)
            st.session_state.processed_images.add(image_path)
        else:
            st.warning("This image has already been processed. Please provide a new image.")

    # Display cumulative summary
    if not st.session_state.nutrition_df.empty:
        st.markdown("### Summary of All Meals")
        st.dataframe(st.session_state.nutrition_df)
        total_calories_today = st.session_state.nutrition_df['Calories (in image)'].sum()
        st.markdown(
            f"""
                    ### Total Calories for Today:  
                    - **Calories Consumed**: {total_calories_today:.2f} kcal  
                    - **Remaining Calories**: {max(daily_calories - total_calories_today, 0):.2f} kcal  
                    """
        )

        total_protein = st.session_state.nutrition_df["Protein (per 100g)"].sum()
        total_fat = st.session_state.nutrition_df["Fat (per 100g)"].sum()
        total_calories = st.session_state.nutrition_df["Calories (in image)"].sum()

        # Display requirements first
        st.sidebar.markdown("### Daily Nutritional Requirements")
        st.sidebar.markdown(
            f"""
                    - **Calories**: {daily_calories:.2f} kcal  
                    - **Protein**: {(daily_calories * 0.2) / 4:.2f} g  
                    - **Fat**: {(daily_calories * 0.3) / 9:.2f} g  
                    """
        )

        # Nutritional Analysis
        st.sidebar.markdown("### Nutrient Analysis")
        if total_calories >= daily_calories:
            st.sidebar.markdown(f"✅ <span style='color:green;'>You have met your calorie requirement!</span>",
                                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                f"⚠️ <span style='color:red;'>You are lacking {daily_calories - total_calories:.2f} kcal.</span>",
                unsafe_allow_html=True)

        if total_protein >= (daily_calories * 0.2) / 4:
            st.sidebar.markdown(f"✅ <span style='color:green;'>You have met your protein requirement!</span>",
                                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                f"⚠️ <span style='color:red;'>You are lacking {((daily_calories * 0.2) / 4) - total_protein:.2f} g of protein.</span>",
                unsafe_allow_html=True)

        if total_fat >= (daily_calories * 0.3) / 9:
            st.sidebar.markdown(f"✅ <span style='color:green;'>You have met your fat requirement!</span>",
                                unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                f"⚠️ <span style='color:red;'>You are lacking {((daily_calories * 0.3) / 9) - total_fat:.2f} g of fat.</span>",
                unsafe_allow_html=True)

    st.markdown("##### For additional images analysis enter new image URL and press the 'Process Image' button")


if st.session_state["page"] == "main":
    main_app()
else:
    st.sidebar.write("Choose an option:")
    option = st.sidebar.radio("Authentication", ["Login", "Sign Up"])
    if option == "Login":
        st.title("Login Page")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", placeholder="Enter your password", type="password")

        if st.button("Log In"):
            # Retrieve user by username from MongoDB
            user = get_user_by_username(username)

            if user and bcrypt.checkpw(password.encode("utf-8"), user["hashed_password"]):
                # Successful login
                st.session_state["logged_in"] = True
                st.session_state["page"] = "main"
                st.session_state["username"] = user["username"]
                st.rerun()
            else:
                st.error("Invalid username or password")
    elif option == "Sign Up":
        sign_up()
