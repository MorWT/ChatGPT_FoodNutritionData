# ChatGPT Food Nutrition Assistant

## Overview

The **ChatGPT Food Nutrition Assistant** is a user-friendly application designed to provide nutritional insights from food images. It leverages advanced language models to identify food items from images, calculate their nutritional content (e.g., calories, fat, protein), and help users track their daily caloric and nutritional intake. The app also includes user authentication and personalized caloric recommendations based on individual profiles.

---

## Features

- **Image-based Food Identification**: Analyze food items in images and retrieve detailed nutritional data.
- **Personalized Nutritional Guidance**: Tailored caloric recommendations based on user input (e.g., weight, height, activity level).
- **Daily Summary**: Track total calories, protein, and fat consumed.
- **User Authentication**: Secure user sign-up and login functionality.

---

## Prerequisites

1. **Python Environment**: Python 3.8 or higher.
2. **MongoDB Database**: A running MongoDB instance with connection details in `create_DB.py`.
3. **Environment Variables**: Set up an `.env` file with your OpenAI API key: OPENAI_API_KEY=<Your OpenAI API Key>
4. **Dependencies**: All required Python packages exist in the poetry environment

---

## How to Run

Follow these steps to set up and run the application:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ChatGPT_FoodNutritionData.git
cd ChatGPT_FoodNutritionData
```

### 2. Install Dependencies
Ensure you have Poetry installed on your system. If not, install it by following the [Poetry Installation Guide](https://python-poetry.org/docs/#installation).

### 3. Set Up Environment Variables
Create a .env file in the root directory and add your OpenAI API key:
```bash
cd ChatGPT_FoodNutritionData
touch .env
OPENAI_API_KEY=<Your OpenAI API Key>
```

### 4. Run the Application
Launch the application using Streamlit:
```bash
poetry shell
streamlit run LLM_ObjectDetection.py
```


