import os
import time
from dotenv import load_dotenv
from groq import Groq  # <-- Correct import

# Load environment variables from .env file
load_dotenv()

free_models = ["gemma2-9b-it", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b", "mistral-saba-24b"]

def get_groq_response(prompt, model=free_models[0]):
    """Get a response from the Groq API."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # <-- Correct initialization

    # Add rate limiting - wait between requests to avoid overwhelming the API
    time.sleep(0.5)  # 500ms delay between requests
    
    # Try all available models if needed
    available_models = free_models.copy()
    current_model = model
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=current_model,
                timeout=30,  # 30 second timeout
            )
            # If successful, return the response
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            error_type = str(type(e).__name__)
            error_msg = str(e)
            print(f"Error ({error_type}) with model {current_model}: {error_msg}")
            
            # Handle rate limiting errors with exponential backoff
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                wait_time = retry_delay * (2 ** attempt) + 5  # Add extra 5 seconds for rate limits
                print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            # Try a different model if available
            if current_model in available_models:
                available_models.remove(current_model)
            
            if available_models:
                current_model = available_models[0]
                print(f"Switching to model: {current_model}")
            else:
                # If we've tried all models, wait and retry the first one
                print("All models attempted. Retrying original model after delay...")
                available_models = free_models.copy()
                current_model = model
                time.sleep(retry_delay * (2 ** attempt))
    
    # If all retries and models fail, return a fallback message
    return "Unable to process."
