from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from database import DBSessionDep
from entities import Conversation
from repositories.conversations import ConversationRepository
from schemas import ConversationCreate, ConversationOut, ConversationUpdate, MessageOut
from services.conversations import ConversationService

router = APIRouter(prefix="/conversations")


async def get_conversation(conversation_id: int, session: DBSessionDep) -> Conversation:
    conversation = await ConversationRepository(session).get(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation


GetConversationDep = Annotated[Conversation, Depends(get_conversation)]


@router.get("")
async def list_conversations_controller(
    session: DBSessionDep, skip: int = 0, take: int = 100
) -> list[ConversationOut]:
    conversations = await ConversationRepository(session).list(skip, take)
    return [ConversationOut.model_validate(c) for c in conversations]


@router.get("/{conversation_id}")
async def get_conversation_controller(
    conversation: GetConversationDep,
) -> ConversationOut:
    return ConversationOut.model_validate(conversation)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_conversation_controller(
    conversation: ConversationCreate, session: DBSessionDep
) -> ConversationOut:
    new_conversation = await ConversationRepository(session).create(conversation)
    return ConversationOut.model_validate(new_conversation)


@router.put("/{conversation_id}", status_code=status.HTTP_202_ACCEPTED)
async def update_conversation_controller(
    conversation: GetConversationDep,
    updated_conversation: ConversationUpdate,
    session: DBSessionDep,
) -> ConversationOut:
    updated = await ConversationRepository(session).update(
        conversation.id, updated_conversation
    )
    return ConversationOut.model_validate(updated)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_controller(
    conversation: GetConversationDep, session: DBSessionDep
) -> None:
    await ConversationRepository(session).delete(conversation.id)


@router.get("/{conversation_id}/messages")
async def list_conversation_messages_controller(
    conversation: GetConversationDep,
    session: DBSessionDep,
) -> list[MessageOut]:
    messages = await ConversationService(session).list_messages(conversation.id)
    return [MessageOut.model_validate(m) for m in messages]
