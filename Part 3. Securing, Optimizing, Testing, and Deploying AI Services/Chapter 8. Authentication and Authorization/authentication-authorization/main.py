from typing import Annotated
import routes
from entities import User
from fastapi import Depends, FastAPI
from services.auth import AuthService
auth_service = AuthService()
AuthenticateUserDep = Annotated[User, Depends(auth_service.get_current_user)]
...
app = FastAPI(lifespan=lifespan)
app.include_router(routes.auth.router, prefix="/auth", tags=["Auth"])
app.include_router(
 routes.resource.router,
 dependencies=[AuthenticateUserDep],
 prefix="/generate",
306 | Chapter 8: Authentication and Authorization
 tags=["Generate"],
)
... # Add other routes to the app here