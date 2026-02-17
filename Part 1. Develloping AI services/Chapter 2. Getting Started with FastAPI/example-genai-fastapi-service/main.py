# main.py
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

# ⚠️ En production, utiliser des variables d'environnement
openai_client = OpenAI(api_key="your_api_key")


@app.get("/")
def root_controller():
    return {"status": "healthy"}


@app.get("/chat")
def chat_controller(prompt: str = "Inspire me"):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    statement = response.choices[0].message.content
    return {"statement": statement}