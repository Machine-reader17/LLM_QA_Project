# LLM_QA_CLI.py
import os
import re
import string
# For Gemini API:
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv



# Load variables from .env file (Does nothing in a hosted environment like Render)
load_dotenv() 

# The key is loaded into the OS environment variables
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Now initialize the client using the key
if GEMINI_KEY:
    try:
        # The genai.Client() constructor automatically looks for the GEMINI_API_KEY
        # in the environment if the key= argument is omitted.
        from google import genai
        llm_client = genai.Client() 
    except Exception as e:
        print(f"Error initializing LLM Client: {e}")
        llm_client = None
else:
    print("FATAL: GEMINI_API_KEY not found in environment variables.")
    llm_client = None
# Set your API Key (e.g., from environment variables)



def preprocess_question(question: str) -> str:
    """Applies basic preprocessing to the input question."""
    # 1. Lowercasing
    processed_q = question.lower()
    
    # 2. Punctuation Removal (replace with a space to avoid joining words)
    processed_q = processed_q.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Basic Cleaning (e.g., excessive whitespace)
    processed_q = re.sub(r'\s+', ' ', processed_q).strip()
    
    return processed_q

def get_llm_answer(question: str, llm_client) -> str:
    """Constructs a prompt and sends it to the LLM API."""
    
    # 1. Construct the Prompt (System instruction for better QA)
    system_instruction = "You are an expert Question-and-Answering system. Provide a concise and accurate answer to the user's question."
    
    # 2. Call the API 
    try:
        response = llm_client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=question,
            config={'system_instruction': system_instruction}
        )
        return response.text
    except APIError as e:
        return f"ERROR: LLM API call failed. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    
def main():
    """Main function for the CLI application."""
    print(" NLP Q&A System - CLI")
    print("-" * 30)
    
    # Initialize the LLM client (assuming API key is set)
    try:
        llm_client = genai.Client()
    except Exception as e:
        print(f"Failed to initialize LLM client. Check your API key. Error: {e}")
        return

    while True:
        try:
            # 1. Accept natural-language question
            user_question = input("\nEnter your question (or type 'quit' or 'exit'):\n> ")
            
            if user_question.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if not user_question.strip():
                continue

            # 2. Apply basic preprocessing
            processed_q = preprocess_question(user_question)
            
            # 3. Construct prompt and send to LLM API
            print(f"\nProcessing question: **'{processed_q}'**")
            print("Thinking...")
            
            final_answer = get_llm_answer(processed_q, llm_client)
            
            # 4. Display the final answer
            print("\n--- LLM ANSWER ---")
            print(final_answer)
            print("------------------")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()