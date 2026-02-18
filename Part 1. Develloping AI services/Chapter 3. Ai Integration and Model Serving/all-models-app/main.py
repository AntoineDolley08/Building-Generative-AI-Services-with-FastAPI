import csv
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

from fastapi import FastAPI, File, Request, Response, status
from fastapi.responses import StreamingResponse
from PIL import Image

from models import (
    generate_3d_geometry,
    generate_audio,
    generate_image,
    generate_text,
    generate_video,
    load_3d_model,
    load_audio_model,
    load_image_model,
    load_text_model,
    load_video_model,
)
from schemas import VoicePresets
from utils import (
    audio_array_to_buffer,
    export_to_video_buffer,
    img_to_bytes,
    mesh_to_obj_buffer,
)

models = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Chargement des modèles au démarrage
    models["text"] = load_text_model()
    models["audio_processor"], models["audio_model"] = load_audio_model()
    models["image"] = load_image_model()
    models["video"] = load_video_model()
    models["3d"] = load_3d_model()
    yield
    # Cleanup
    models.clear()


app = FastAPI(lifespan=lifespan)

csv_header = [
    "Request ID",
    "Datetime",
    "Endpoint Triggered",
    "Client IP Address",
    "Response Time",
    "Status Code",
    "Successful",
]


@app.middleware("http")
async def monitor_service(
    req: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = uuid4().hex
    request_datetime = datetime.now(UTC).isoformat()
    start_time = time.perf_counter()
    response: Response = await call_next(req)
    response_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-Response-Time"] = str(response_time)
    response.headers["X-API-Request-ID"] = request_id
    with open("usage.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(csv_header)
        writer.writerow(
            [
                request_id,
                request_datetime,
                req.url,
                req.client.host,
                response_time,
                response.status_code,
                response.status_code < 400,
            ]
        )
    return response


@app.get("/generate/text")
def serve_language_model_controller(prompt: str) -> str:
    output = generate_text(models["text"], prompt)
    return output


# Fast Api Passtrough for BentoML
# @app.get(
#     "/generate/bentoml/image",
#     responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
#     response_class=Response,
# )
# async def serve_bentoml_text_to_image_controller(prompt: str):
#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             "http://localhost:5000/generate", json={"prompt": prompt}
#         )
#     return Response(content=response.content, media_type="image/png")


@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_audio_model_controller(
    prompt: str,
    preset: VoicePresets = "v2/en_speaker_1",
):
    output, sample_rate = generate_audio(
        models["audio_processor"], models["audio_model"], prompt, preset
    )
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate), media_type="audio/wav"
    )


@app.get(
    "/generate/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
    response_class=Response,
)
def serve_text_to_image_model_controller(prompt: str):
    output = generate_image(models["image"], prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")


@app.post(
    "/generate/video",
    responses={status.HTTP_200_OK: {"content": {"video/mp4": {}}}},
    response_class=StreamingResponse,
)
def serve_image_to_video_model_controller(
    image: bytes = File(...), num_frames: int = 25
):
    image = Image.open(BytesIO(image))
    frames = generate_video(models["video"], image, num_frames)
    return StreamingResponse(export_to_video_buffer(frames), media_type="video/mp4")


@app.get(
    "/generate/3d",
    responses={status.HTTP_200_OK: {"content": {"model/obj": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_3d_model_controller(prompt: str, num_inference_steps: int = 25):
    mesh = generate_3d_geometry(models["3d"], prompt, num_inference_steps)
    response = StreamingResponse(mesh_to_obj_buffer(mesh), media_type="model/obj")
    response.headers["Content-Disposition"] = f"attachment; filename={prompt}.obj"
    return response
