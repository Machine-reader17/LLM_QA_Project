# app.py
from flask import Flask, render_template, request
import os
# Import/replicate the functions from LLM_QA_CLI.py
from LLM_QA_CLI import preprocess_question, get_llm_answer 
# Assuming you set up the LLM_QA_CLI.py to be importable and initialized the client there or here
from google import genai 

app = Flask(__name__)

# Only needed for local testing:
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

@app.route('/', methods=['GET', 'POST'])
def index():
    user_question = ""
    processed_question = ""
    llm_response = "Enter a question above to get an answer."

    if request.method == 'POST':
        user_question = request.form.get('question', '')
        
        if user_question:
            # 1. View the processed question
            processed_question = preprocess_question(user_question)
            
            # 2. See the LLM API response (and display generated answer)
            if llm_client:
                llm_response = get_llm_answer(processed_question, llm_client)
            else:
                llm_response = "ERROR: The LLM service is currently unavailable. Check the API key configuration."
        else:
            llm_response = "Please enter a non-empty question."

    # Render the HTML template, passing data to it
    return render_template(
        'index.html',
        user_question=user_question,
        processed_question=processed_question,
        llm_response=llm_response
    )

if __name__ == '__main__':
    # Use a port suitable for deployment (e.g., 5000 is default, but others might be needed)
    app.run(debug=True)