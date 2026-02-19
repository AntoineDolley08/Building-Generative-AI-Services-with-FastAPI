from fastapi import APIRouter
router = APIRouter(prefix="/generate", tags=["Resource"])
@router.get("/generate/text", ...)
def serve_language_model_controller(...):
 ...
@router.get("/generate/audio", ...)
def serve_text_to_audio_model_controller(...)
 ...
... # Add other controllers to the resource router here