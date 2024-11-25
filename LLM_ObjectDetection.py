import os
import base64
import requests
import streamlit as st
import pandas as pd
from urllib.parse import urlparse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image
import json

OPENAI_API_KEY = ('sk-proj-zRcdPL6FzeYco9zbuJA7IAGLRGUKpLDwA8mWE170y1tQbqdAxTAQnIy69LIN4chYjyTUMGzWA9T3BlbkFJySLP3EvBqq'
                  'PalucM9_nLTD6m_q_465dK9ArhMDcr2Z1DCaD5KrgrRbTkdE1_RK8H2rJUFQKeEA')


class NutritionData(BaseModel):
    name: str = Field(description="The name of a food item")
    calories_per_100g: float = Field(description="The calories value of the food item per 100g")
    fat_per_100g: float = Field(description="The fat value of the food of the food item per 100g")
    protein_per_100g: float = Field(description="The protein value of  the food item per 100g")
    weight_grams: float = Field(description="The estimated weight of the food item in grams")


class ImageRecognitionResult(BaseModel):
    food_items: list[NutritionData]


# Step 1: Check if the image_path is a URL
def is_url(path):
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def download_image(image_path):
    # Step 2: Download the image
    response = requests.get(image_path, stream=True)
    if response.status_code == 200:
        local_filename = "downloaded_image.jpg"  # Save the image with a local filename
        with open(local_filename, "wb") as img_file:
            for chunk in response.iter_content(chunk_size=8192):
                img_file.write(chunk)
        image_path = local_filename  # Update image_path to the local file
    else:
        raise Exception(f"Failed to download image from URL: {image_path}")

    return image_path


def resize_image_to_max_500px(image_path):
    """
    Resizes an image to ensure the maximum side is 500 pixels while maintaining the aspect ratio.
    If the image is already within the size limit, it returns the original path.
    Otherwise, it saves and returns the path to the resized image.

    Args:
        image_path (str): The path to the input image.

    Returns:
        str: The path to the resized image or the original path if resizing was not needed.
    """
    with Image.open(image_path) as img:
        max_side = max(img.size)

        # Check if the image exceeds the size limit
        if max_side <= 500:
            return image_path  # Return original path if resizing is not needed

        # Calculate the resize ratio to fit the maximum side to 500px
        ratio = 500 / max_side
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))

        # Resize the image
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)  # Use the updated resampling method

        # Save the resized image to a new file
        resized_image_path = "resized_" + os.path.basename(image_path)
        resized_img.save(resized_image_path)

        return resized_image_path  # Return the path to the resized image


# Define the parser using NutritionData
parser = PydanticOutputParser(pydantic_object=NutritionData)

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)


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
                   "Once identified, you will provide nutritional details such as name, calories, fat, and protein per "
                   "100g. Base64-encoded images will be provided"),
        human_message
    ])

    # Step 3: Create a chain and pass the Base64 image
    chain = prompt | llm.with_structured_output(ImageRecognitionResult)
    result = chain.invoke({
        "image_b64": image_b64  # Pass the Base64-encoded image string here
    })

    return result.model_dump_json(indent=2)


if __name__ == "__main__":
    counter = 0
    st.title("Food Nutrition Assistant")
    image_path = st.text_input("Provide a link to an image", key=f"image_path_{counter}")

    if image_path:
        try:
            st.image(image_path, caption="Uploaded Image", use_column_width=True)
            st.write("Processing the image...")
            result = process_image(image_path)

            # Parse the JSON string into a dictionary
            result_dict = json.loads(result)

            # Convert dictionary to ImageRecognitionResult object
            result_data = ImageRecognitionResult(**result_dict)

            food_calories = []
            total_calories = 0

            # Perform calculations for each food item
            for item in result_data.food_items:
                item_calories = (item.calories_per_100g / 100) * item.weight_grams
                total_calories += item_calories
                food_calories.append({
                    "Food Name": item.name.capitalize(),
                    "Calories (per 100g)": item.calories_per_100g,
                    "Fat (per 100g)": item.fat_per_100g,
                    "Protein (per 100g)": item.protein_per_100g,
                    "Weight (grams)": item.weight_grams,
                    "Calories (in image)": round(item_calories, 2)
                })


            # Convert to DataFrame for display
            df = pd.DataFrame(food_calories)

            # Display the table in Streamlit
            st.markdown("### Calories Calculation Table")
            st.dataframe(df)

            # Display the total calories
            st.markdown(f"### Total Calories in Image: {round(total_calories, 2)}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")