from fastapi.security import HTTPBearer
from passlib.context import CryptContext
class PasswordService:
 security = HTTPBearer()
 pwd_context = CryptContext(schemes=["bcrypt"])
 async def verify_password(
Authentication Methods | 299
 self, password: str, hashed_password: str
 ) -> bool:
 return self.pwd_context.verify(password, hashed_password)
 async def get_password_hash(self, password: str) -> str:
 return self.pwd_context.hash(password)

from datetime import UTC, datetime, timedelta
from exceptions import UnauthorizedException
from jose import JWTError, jwt
from pydantic import UUID4
from repositories import TokenRepository
from schemas import TokenCreate, TokenUpdate
class TokenService(TokenRepository):
 secret_key = "your_secret_key"
 algorithm = "HS256"
 expires_in_minutes = 60
async def create_access_token(
 self, data: dict, expires_delta: timedelta | None = None
) -> str:
 to_encode = data.copy()
 if expires_delta:
 expire = datetime.now(UTC) + expires_delta
 else:
 expire = datetime.now(UTC) + timedelta(minutes=self.expires_in_minutes)
 token_id = await self.create(TokenCreate(expires_at=expire))
 to_encode.update(
 {"exp": expire, "iss": "your_service_name", "sub": token_id}
 )
 encoded_jwt = jwt.encode(
 to_encode, self.secret_key, algorithm=self.algorithm
 )
 return encoded_jwt
async def deactivate(self, token_id: UUID4) -> None:
 await self.update(TokenUpdate(id=token_id, is_active=False))
302 | Chapter 8: Authentication and Authorization
def decode(self, encoded_token: str) -> dict:
 try:
 return jwt.decode(
 encoded_token, self.secret_key, algorithms=[self.algorithm]
 )
 except JWTError:
 raise UnauthorizedException
async def validate(self, token_id: UUID4) -> bool:
 return (token := await self.get(token_id)) is not None and token.is_active

from typing import Annotated
from databases import DBSessionDep
from entities import Token, User, UserCreate, UserInDB
from exceptions import AlreadyRegisteredException, UnauthorizedException
from fastapi import Depends
from fastapi.security import (HTTPAuthorizationCredentials, HTTPBearer,
 OAuth2PasswordRequestForm)
from services.auth import PasswordService, TokenService
from services.users import UserService
security = HTTPBearer()
LoginFormDep = Annotated[OAuth2PasswordRequestForm, Depends()]
Authentication Methods | 303
AuthHeaderDep = Annotated[HTTPAuthorizationCredentials, Depends(security)]
class AuthService:
 def __init__(self, session: DBSessionDep):
 self.password_service = PasswordService()
 self.token_service = TokenService(session)
 self.user_service = UserService(session)
 async def register_user(self, user: UserCreate) -> User:
 if await self.user_service.get(user.username):
 raise AlreadyRegisteredException
 hashed_password = await self.password_service.get_password_hash(
 user.password
 )
 return await self.user_service.create(
 UserInDB(username=user.username, hashed_password=hashed_password)
 )
 async def authenticate_user(self, form_data: LoginFormDep) -> Token:
 if not (user := await self.user_service.get_user(form_data.username)):
 raise UnauthorizedException
 if not await self.password_service.verify_password(
 form_data.password, user.hashed_password
 ):
 raise UnauthorizedException
 return await self.token_service.create_access_token(user._asdict())
 async def get_current_user(self, credentials: AuthHeaderDep) -> User:
 if credentials.scheme != "Bearer":
 raise UnauthorizedException
 if not (token := credentials.credentials):
 raise UnauthorizedException
 payload = self.token_service.decode(token)
 if not await self.token_service.validate(payload.get("sub")):
 raise UnauthorizedException
 if not (username := payload.get("username")):
 raise UnauthorizedException
 if not (user := await self.user_service.get(username)):
 raise UnauthorizedException
 return user
 async def logout(self, credentials: AuthHeaderDep) -> None:
 payload = self.token_service.decode(credentials.credentials)
 await self.token_service.deactivate(payload.get("sub"))
 # Add Password Reset Method
 async def reset_password(self): ...