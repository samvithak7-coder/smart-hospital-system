import os
from groq import Groq

def get_groq_client(api_key: str = None) -> Groq:
    """
    Returns an instance of the official Groq client using the GROQ_API_KEY environment variable.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    return Groq(api_key=key)

def generate_report(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Generate medical insights or reports using Groq Python SDK with model llama-3.1-8b-instant.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are Kvasir-HealthAI, a medical AI assistant specializing in gastrointestinal polyp segmentation and report analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model
    )
    return response.choices[0].message.content
