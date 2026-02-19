from entities import Message
from repositories.interfaces import Repository
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class MessageRepository(Repository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, conversation_id: int) -> list[Message]:
        result = await self.session.execute(
            select(Message).where(Message.conversation_id == conversation_id)
        )
        return [m for m in result.scalars().all()]

    async def get(self, message_id: int) -> Message | None:
        result = await self.session.execute(
            select(Message).where(Message.id == message_id)
        )
        return result.scalars().first()

    async def create(self, record: Message) -> Message:
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def update(self, uid: int, record: Message) -> Message | None:
        message = await self.get(uid)
        if not message:
            return None
        for key, value in vars(record).items():
            if not key.startswith("_"):
                setattr(message, key, value)
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def delete(self, message_id: int) -> None:
        message = await self.get(message_id)
        if not message:
            return
        await self.session.delete(message)
        await self.session.commit()
