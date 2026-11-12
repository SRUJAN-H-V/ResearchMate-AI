
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Read API key and model name
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME", "gemini-3.8-flash")

if not api_key or api_key == "your_google_api_key_here":
    raise ValueError("Please configure GOOGLE_API_KEY in your .env file")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    temperature=0.3,
    timeout=120,
    max_retries=2
)
def test_gemini():
    response = llm.invoke(
        "Explain what a research paper is in two sentences."
    )
    return response.text
