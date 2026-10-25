
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langfuse import Langfuse

# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LANGFUSE_PUBLIC_KEY = os.getenv(
    "LANGFUSE_PUBLIC_KEY", ""
).strip()
LANGFUSE_SECRET_KEY = os.getenv(
    "LANGFUSE_SECRET_KEY", ""
).strip()
LANGFUSE_BASE_URL = os.getenv(
    "LANGFUSE_BASE_URL",
    "https://us.cloud.langfuse.com"
).strip()

# ==========================================
# 2. VALIDATE API KEYS
# ==========================================

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env")

if not LANGFUSE_PUBLIC_KEY:
    raise ValueError("LANGFUSE_PUBLIC_KEY is missing in .env")

if not LANGFUSE_SECRET_KEY:
    raise ValueError("LANGFUSE_SECRET_KEY is missing in .env")

# ==========================================
# 3. CONFIGURATION
# ==========================================

MODEL_NAME = "gemini-3.8-flash"

STUDY_MODES = {
    "1": {
        "name": "Python Tutor",
        "instruction": (
            "Teach Python programming using simple explanations, "
            "beginner-friendly examples, and runnable code. "
            "Explain errors and debugging step by step."
        )
    },
    "2": {
        "name": "SQL Tutor",
        "instruction": (
            "Teach SQL, including SELECT, WHERE, GROUP BY, "
            "functions, joins, subqueries, and database concepts. "
            "Provide example queries and explain their output."
        )
    },
    "3": {
        "name": "Machine Learning Tutor",
        "instruction": (
            "Teach machine learning algorithms, data preprocessing, "
            "model training, evaluation, and Python examples. "
            "Explain technical terms clearly."
        )
    },
    "4": {
        "name": "General AI Tutor",
        "instruction": (
            "Teach artificial intelligence, LLMs, prompt engineering, "
            "RAG, embeddings, and AI application development "
            "using clear explanations and practical examples."
        )
    }
}

# ==========================================
# 4. INITIALIZE CLIENTS
# ==========================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    base_url=LANGFUSE_BASE_URL
)

# ==========================================
# 5. STUDY MODE SELECTION
# ==========================================

def choose_mode():
    print("\n========== SELECT STUDY MODE ==========")

    for number, mode in STUDY_MODES.items():
        print(f"{number}. {mode['name']}")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice in STUDY_MODES:
            return STUDY_MODES[choice]

        print("Invalid choice. Please enter a number from 1 to 4.")


# ==========================================
# 6. CREATE CHAT SESSION
# ==========================================

def create_chat(mode):
    return client.chats.create(
        model=MODEL_NAME,
        config={
            "system_instruction": (
                "You are a helpful AI Study Assistant. "
                + mode["instruction"]
                + " Be accurate, organized, and easy to understand. "
                "Use code examples when appropriate. "
                "If you are uncertain, say so instead of inventing facts."
            )
        }
    )


# ==========================================
# 7. SEND QUESTION AND TRACK IN LANGFUSE
# ==========================================

def ask_assistant(chat, question, mode_name):

    with langfuse.start_as_current_observation(
        as_type="span",
        name="ai-study-assistant",
        input={
            "question": question,
            "study_mode": mode_name
        }
    ) as span:

        response = chat.send_message(question)

        answer = response.text or "No response received."

        span.update(
            output={
                "answer": answer,
                "study_mode": mode_name
            }
        )

    return answer


# ==========================================
# 8. MAIN APPLICATION
# ==========================================

def main():
    try:
        print("\n======================================")
        print("       AI STUDY ASSISTANT")
        print("      Powered by Gemini + Langfuse")
        print("======================================")

        mode = choose_mode()
        chat = create_chat(mode)

        print(f"\nSelected mode: {mode['name']}")
        print("\nCommands:")
        print("  mode  - Change study mode")
        print("  exit  - Close the application")
        print("\nStart asking your questions!\n")

        while True:
            try:
                question = input("You: ").strip()

                if question.lower() == "exit":
                    break

                if not question:
                    continue

                # Change study mode
                if question.lower() == "mode":
                    mode = choose_mode()
                    chat = create_chat(mode)

                    print(f"\nSwitched to: {mode['name']}")
                    print("A new conversation has started.\n")
                    continue

                # Get answer and track interaction
                answer = ask_assistant(
                    chat,
                    question,
                    mode["name"]
                )

                print(f"\nAssistant ({mode['name']}):")
                print(answer)
                print()

            except KeyboardInterrupt:
                print("\n\nApplication interrupted.")
                break

            except Exception as error:
                print("\nRequest failed:", type(error).__name__)
                print("Details:", str(error))
                print("Please check your API configuration and connection.\n")

    finally:
        try:
            langfuse.flush()
            print("Langfuse events flushed.")
        except Exception as error:
            print("Langfuse flush failed:", str(error))

        try:
            client.close()
        except Exception:
            pass

        print("AI Study Assistant stopped.")


# ==========================================
# 9. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
