from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import engine
from routers.conversations import router as conversations_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.include_router(conversations_router)
