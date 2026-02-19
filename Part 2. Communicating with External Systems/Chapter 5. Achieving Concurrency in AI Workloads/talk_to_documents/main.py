import os

from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
    BackgroundTasks,
)
from openai import AsyncOpenAI, OpenAI
from typing import Annotated
from schemas import TextModelRequest, TextModelResponse
from fastapi import Depends, Request
from rag import generate_text, models

from dependencies import get_rag_content, get_urls_content


from rag import pdf_text_extractor, vector_service

from upload import save_file

app = FastAPI()
sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.post("/sync")
def sync_generate_text(prompt: str = Body(...)):
    completion = sync_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content


@app.post("/async")
async def async_generate_text(prompt: str = Body(...)):
    completion = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content


@app.post("/upload")
async def file_upload_controller(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
    bg_text_processor: BackgroundTasks,
):
    try:
        filepath = await save_file(file)
        bg_text_processor.add_task(pdf_text_extractor, filepath)
        bg_text_processor.add_task(
            vector_service.store_file_content_in_db,
            filepath.replace("pdf", "txt"),
            512,
            "knowledgebase",
            768,
        )
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}


@app.post("/generate/text", response_model_exclude_defaults=True)
async def serve_text_to_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
    rag_content: str = Depends(get_rag_content),
) -> TextModelResponse:
    # Raise HTTPException for invalid models
    prompt = body.prompt + " " + urls_content + rag_content
    output = generate_text(models["text"], prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
