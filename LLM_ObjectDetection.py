import os
import requests
import streamlit as st
from PIL import Image
from base64 import b64encode
from io import BytesIO
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

MAX_TOKENS = 100000

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)


# Define the NutritionData class for structured output
class NutritionData(BaseModel):
    name: str = Field(description="The name of the food shown in the image")
    calories: float = Field(description="The calories value of the food shown in the image for 100g")
    fat: float = Field(description="The fat value of the food shown in the image for 100g")
    protein: float = Field(description="The protein value of the food shown in the image for 100g")


# Define the parser using NutritionData
parser = PydanticOutputParser(pydantic_object=NutritionData)


# Function to process and re-encode the image as Base64
def process_image(image_source, source_type, max_size=(512, 512)):
    try:
        # Load the image from the file or URL
        if source_type == "upload":
            image = Image.open(image_source)
        elif source_type == "url":
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))

        # Resize the image to reduce size
        image = image.convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing

        # Convert to Base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image_b64 = b64encode(image_data).decode("utf-8")
        return image, image_b64
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None, None


def validate_token_size(image_b64):
    if len(image_b64) > MAX_TOKENS:
        st.error("The encoded image is too large to process. Please use a smaller image.")
        return False
    return True


# Function to analyze the image using GPT-4o
def analyze_image(image_b64):
    # Construct the prompt for GPT-4o
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", f"You are an assistant that detects food items from images. "
    #                f"Given an image, infer the type of food shown and provide detailed nutritional information "
    #                f"such as name, calories, fat, and protein per 100g. "
    #                f"Use the following format: {format_instructions}."),
    #     ("human", f"Here is the image: data:image/jpeg;base64,{image_b64}")
    # ])
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert nutrition assistant that specializes in identifying food items from images. "
                   f"Given an image of food, you will analyze it to identify the type of food. Once identified, "
                   f"you will provide nutritional details such as name, calories, fat, and protein per 100g. "
                   f"Base64-encoded images will be provided, and your response must follow this format: {format_instructions}."),
        ("human", f"Here is the image: data:image/jpeg;base64,{image_b64}")
    ])

    # Run the chain
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "language": "English",
            "format_instructions": format_instructions
        })
        return result
    except Exception as e:
        return {"error": str(e)}


# Streamlit interface
if __name__ == "__main__":
    st.title("Food Nutrition Assistant")

    # Input options
    st.subheader("Input Options:")
    upload_flag = st.checkbox("Upload an image from your computer")
    url_flag = st.checkbox("Provide a URL to an image")

    image = None
    image_b64 = None

    # Process uploaded image
    if upload_flag:
        uploaded_file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image, image_b64 = process_image(uploaded_file, source_type="upload")

    # Process URL image
    if url_flag:
        image_url = st.text_input("Enter the image URL")
        if image_url:
            image, image_b64 = process_image(image_url, source_type="url")

    # Analyze and display results
    if image and image_b64:
        if validate_token_size(image_b64):
            st.image(image, caption="Processed Image", use_column_width=True)
            st.subheader("Nutritional Information:")
            response = analyze_image(image_b64)

            # Display the results
            if "error" in response:
                st.error("The image could not be processed. Please try a smaller image or describe the food item.")

            elif isinstance(response, NutritionData):
                st.write(f"**Food Name:** {response.name}")
                st.write(f"**Calories (per 100g):** {response.calories}")
                st.write(f"**Fat (per 100g):** {response.fat}")
                st.write(f"**Protein (per 100g):** {response.protein}")
            else:
                st.error("Unexpected response format.")
