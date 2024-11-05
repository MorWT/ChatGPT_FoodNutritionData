import openai
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser
import tkinter as tk
from tkinter import filedialog, Label, Button
import io
import os

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"======== {OPENAI_API_KEY}")

# Initialize the language model
llm = ChatOpenAI(model='gpt-4o', api_key=OPENAI_API_KEY)

# Define the system prompt for the assistant to provide nutritional information
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an assistant that provides nutritional information about food items. When a food item is specified, "
    "provide details including calories, fat content, protein, vitamins, and other relevant details."
)


# Function to recognize food and get nutritional information
def generate_response(user_input):
    llm.invoke
    prompt = ChatPromptTemplate([
        system_prompt,
        HumanMessagePromptTemplate.from_template("{user_input}")
    ])

    # Run the prompt chain and parse output as string
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": user_input})


# Function to open an image file and process it
def open_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    if file_path:
        # Load the selected image and display it
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Resize for display purposes
        img_display = ImageTk.PhotoImage(image)
        img_label.config(image=img_display)
        img_label.image = img_display

        # Process the image and get nutritional information
        with open(file_path, "rb") as image_file:
            image_data = image_file.read()
            try:
                response = generate_response(image_data)
                response_label.config(text=f"Nutritional Information:\n{response}")
            except Exception as e:
                response_label.config(text=f"Error: {e}")


# Set up the Tkinter GUI
root = tk.Tk()
root.title("Food Nutrition Assistant")
root.geometry("400x500")

# Label to display the selected image
img_label = Label(root)
img_label.pack(pady=10)

# Button to open and upload an image
upload_btn = Button(root, text="Upload Image", command=open_image)
upload_btn.pack(pady=5)

# Label to display the nutritional information
response_label = Label(root, text="Nutritional Information will appear here", wraplength=300, justify="left")
response_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
