
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("MODEL_NAME", "gemini-3.8-flash")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

if not api_key or api_key == "your_google_api_key_here":
    raise ValueError(
        "Please configure GOOGLE_API_KEY in your .env file"
    )


# ==================================================
# 2. INITIALIZE GEMINI (PRIMARY MODEL)
# ==================================================

gemini_llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    timeout=120,
    max_retries=1
)


# ==================================================
# 3. INITIALIZE OLLAMA (LOCAL FALLBACK)
# ==================================================

ollama_llm = ChatOllama(
    model=ollama_model,
    temperature=0
)


# ==================================================
# 4. PROVIDER TRACKING
# ==================================================

last_used_provider = "Not used yet"
providers_used = []


def clear_used_providers():
    """Clear provider history before a new task."""

    global last_used_provider

    providers_used.clear()
    last_used_provider = "Not used yet"


def get_last_used_provider():
    """Return the most recently used model provider."""

    return last_used_provider


def get_used_providers():
    """Return unique providers used during a task."""

    return list(dict.fromkeys(providers_used))


# ==================================================
# 5. DETECT GEMINI ERRORS
# ==================================================

def should_fallback(error):
    """
    Check whether a Gemini error should trigger
    fallback to Ollama.
    """

    status_codes = []

    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)

        if value is not None:
            try:
                status_codes.append(int(value))
            except (TypeError, ValueError):
                pass

    # Temporary API failures and quota errors
    if any(
        code in (429, 500, 502, 503, 504)
        for code in status_codes
    ):
        return True

    error_name = type(error).__name__.lower()
    message = str(error).lower()

    temporary_error_names = (
        "resourceexhausted",
        "serviceunavailable",
        "deadlineexceeded",
        "internalservererror",
        "toomanyrequests",
    )

    if any(
        name in error_name
        for name in temporary_error_names
    ):
        return True

    temporary_messages = (
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "too many requests",
        "service unavailable",
        "deadline exceeded",
        "timed out",
        "timeout",
        "high demand",
        "unavailable",
    )

    return any(
        term in message
        for term in temporary_messages
    )


# ==================================================
# 6. HYBRID MODEL EXECUTION
# ==================================================

def hybrid_invoke(prompt):
    """
    Try Gemini first.

    If Gemini encounters a recognized temporary
    or quota-related failure, use local Ollama.
    """

    global last_used_provider

    try:
        response = gemini_llm.invoke(prompt)

        last_used_provider = "Gemini"
        providers_used.append("Gemini")

        return response

    except Exception as error:

        # Do not hide unrelated programming or
        # configuration errors.
        if not should_fallback(error):
            raise

        print(
            f"\nGemini failed: {error}"
            "\nSwitching to local Ollama model..."
        )

        try:
            response = ollama_llm.invoke(prompt)

            last_used_provider = "Ollama"
            providers_used.append("Ollama")

            print("Ollama fallback successful.")

            return response

        except Exception as ollama_error:

            raise RuntimeError(
                "Both Gemini and Ollama failed.\n"
                f"Gemini error: {error}\n"
                f"Ollama error: {ollama_error}"
            ) from ollama_error


# ==================================================
# 7. SHARED LANGCHAIN INTERFACE
# ==================================================

# Existing agents can continue using:
# from core.llm import llm

llm = RunnableLambda(hybrid_invoke)


# ==================================================
# 8. RESPONSE TEXT EXTRACTION
# ==================================================

def extract_response_text(response):
    """
    Convert LangChain response content or an
    ordinary Python string into readable text.
    """

    # Case 1: Response is already a string.
    if isinstance(response, str):
        return response

    # Case 2: LangChain AIMessage response.
    content = response.content

    if isinstance(content, str):
        return content

    # Case 3: List of content blocks.
    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(
                        block.get("text", "")
                    )

        return "\n".join(
            part for part in text_parts if part
        )

    return str(content)


# ==================================================
# 9. GEMINI-ONLY TEST
# ==================================================

def test_gemini():
    """
    Test Gemini directly without fallback.
    """

    response = gemini_llm.invoke(
        "Explain what a research paper is "
        "in two sentences."
    )

    return extract_response_text(response)


# ==================================================
# 10. HYBRID MODEL TEST
# ==================================================

def test_hybrid():
    """
    Test the Gemini + Ollama model manager.
    """

    clear_used_providers()

    response = llm.invoke(
        "Explain what a research paper is "
        "in two sentences."
    )

    return {
        "provider": get_last_used_provider(),
        "providers_used": get_used_providers(),
        "response": extract_response_text(response)
    }
