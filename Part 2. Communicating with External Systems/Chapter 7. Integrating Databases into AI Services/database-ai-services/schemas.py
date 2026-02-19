from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    title: str
    model_type: str


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(ConversationBase):
    pass


class ConversationOut(ConversationBase):
    id: int
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    conversation_id: int
    prompt_content: str
    response_content: str
    prompt_tokens: int | None
    response_tokens: int | None
    total_tokens: int | None
    is_success: bool | None
    status_code: int | None
    created_at: datetime
    updated_at: datetime
