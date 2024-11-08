import os
import httpx
import requests
import streamlit as st
from PIL import Image
from base64 import b64encode
from io import BytesIO
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


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


# Function to recognize food and get nutritional information
def generate_response(image_data, upload_flag, url_flag, image_url):
    # Convert image bytes to Base64
    if upload_flag:
        image_b64 = b64encode(image_data).decode('utf-8')

    elif url_flag:
        image_b64 = b64encode(httpx.get(image_url).content).decode("utf-8")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that provides nutritional information about food items.\n For given image of "
                   "food, analyze it and provide details such as name, calories, fat, and protein per 100g.\n"
                   "Use the following format: '{format_instructions}'\n"),
        ("human", [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
            },
        ]),
    ])

    # Run the prompt chain and return parsed output
    try:
        chain = prompt | llm | parser
        result = chain.invoke({
            "language": "English",
            "format_instructions": parser.get_format_instructions(),
            "image_data": image_b64
        })
        # Add the parser for structured output
        # result = chain.invoke({"image_b64": image_b64})
        return result
    except Exception as e:
        return f"Error: {e}"


# Streamlit interface
if __name__ == "__main__":
    url_flag = False
    upload_flag = False

    st.title("Food Nutrition Assistant")

    # Option to upload an image
    uploaded_file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])

    # Option to provide an image URL
    image_url = st.text_input("Or provide a link to an image")

    if uploaded_file or image_url:
        try:
            # Load the image from file or URL
            if uploaded_file:
                image = Image.open(uploaded_file)
                image_data = uploaded_file.read()
                st.image(image, caption="Uploaded Image", use_column_width=True)
                upload_flag = True
                image_url = False

            else:
                response = requests.get(image_url)
                response.raise_for_status()  # Ensure the URL is valid
                image_data = BytesIO(response.content).read()
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Image from URL", use_column_width=True)
                upload_flag = True

            # Get nutritional information
            st.subheader("Nutritional Information:")
            response = generate_response(image_data, upload_flag, url_flag, image_url)
            st.write(response)

        except Exception as e:
            st.error(f"Error processing the image: {e}")
