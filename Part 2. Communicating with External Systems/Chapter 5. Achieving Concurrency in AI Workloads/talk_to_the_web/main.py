from fastapi import Body, Depends, Request
from schemas import TextModelResponse
import os

from fastapi import FastAPI
from openai import AsyncOpenAI, OpenAI

app = FastAPI()
sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.post("/generate/text", response_model_exclude_defaults=True)
async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
) -> TextModelResponse:
    ...  # rest of controller logic
    prompt = body.prompt + " " + urls_content
    output = generate_text(models["text"], prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)


@app.post("/sync")
def sync_generate_text(prompt: str = Body(...)):
    completion = sync_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content


@app.post("/async")
async def async_generate_text(prompt: str = Body(...)):
    completion = await async_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content
