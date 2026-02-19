from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from settings import AppSettings

settings = AppSettings()

engine = create_async_engine(settings.database_url, echo=True)

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False, autoflush=False
)


async def get_db_session():
    async with async_session() as session:
        yield session


DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
