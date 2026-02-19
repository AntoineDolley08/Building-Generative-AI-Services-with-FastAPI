# Chapter 8 — Authentication & Authorization

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Authentication vs Authorization](#1-authentication-vs-authorization)
2. [Basic Authentication](#2-basic-authentication)
3. [JWT Authentication](#3-jwt-authentication)
4. [OAuth2 avec GitHub](#4-oauth2-avec-github)
5. [Authorization Models](#5-authorization-models)
6. [RBAC — Role-Based Access Control](#6-rbac--role-based-access-control)
7. [ReBAC — Relationship-Based Access Control](#7-rebac--relationship-based-access-control)
8. [ABAC — Attribute-Based Access Control](#8-abac--attribute-based-access-control)
9. [Hybrid Authorization + Service externe](#9-hybrid-authorization--service-externe)

---

## 1. Authentication vs Authorization

| Concept | Définition | Analogie aéroport |
|---------|-----------|-------------------|
| **Authentication** (AuthN) | Vérifier **qui** tu es | Présenter ton passeport |
| **Authorization** (AuthZ) | Vérifier **ce que** tu peux faire | Avoir le bon visa |

### 4 méthodes d'authentification

| Méthode | Sécurité | Complexité | Use cases |
|---------|----------|------------|-----------|
| **Basic** | ⚠️ Plain text | Très simple | Prototypes, env internes |
| **JWT** | ✅ Signé/encrypté | Moyenne | SPAs, mobiles, REST APIs, microservices |
| **OAuth2** | ✅ Délégué à un IDP | Élevée | SSO, enterprise, accès ressources externes |
| **Key-based** | ✅ Type SSH | Élevée | Petites apps, env internes |

---

## 2. Basic Authentication

Client envoie `Authorization: Basic base64(username:password)` à chaque requête.

```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def authenticate_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> str:
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("UTF-8"), b"ali"
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("UTF-8"), b"secretpassword"
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401, detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

AuthenticatedUserDep = Annotated[str, Depends(authenticate_user)]
```

Points clés :
- `secrets.compare_digest()` → temps constant (protège contre les **timing attacks**)
- Message d'erreur **générique** → ne pas leaker d'info ("Incorrect credentials" pas "User not found")
- ⚠️ **Prototypage uniquement** — credentials en clair à chaque requête

---

## 3. JWT Authentication

### Structure d'un JWT

```
eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWxpIn0.signature
|___ Header ___|.___ Payload ___|._ Signature _|
```

| Partie | Contenu |
|--------|---------|
| **Header** | Type token, algorithme signature |
| **Payload** | Claims (user ID, role, exp, issuer) |
| **Signature** | HMAC(header + payload, secret) |

### Dépendances

```bash
uv add passlib python-jose bcrypt
```

### Architecture

```
[Register] → hash password → store user
[Login]    → verify password → generate JWT → store token → return JWT
[Request]  → Bearer token → decode → validate → return user
[Logout]   → decode → deactivate token
```

### ORM Models (User + Token)

```python
# entities.py
class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True)
    role: Mapped[str] = mapped_column(default="USER")
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )
    tokens = relationship("Token", back_populates="user", cascade="all, delete-orphan")
    __table_args__ = (Index("ix_users_email", "email"),)

class Token(Base):
    __tablename__ = "tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime] = mapped_column()
    is_active: Mapped[bool] = mapped_column(default=True)
    ip_address: Mapped[str | None] = mapped_column(String(255))
    user = relationship("User", back_populates="tokens")
    __table_args__ = (
        Index("ix_tokens_user_id", "user_id"),
        Index("ix_tokens_ip_address", "ip_address"),
    )
```

Décisions sécurité : UUID (IDs non devinables), `hashed_password` (jamais en clair), `is_active` (désactivation possible), `role` (préparation RBAC), index sur email/user_id/ip_address.

### Pydantic Schemas avec validation

```python
# schemas.py
ValidUsername = Annotated[str, Field(min_length=3, max_length=20), AfterValidator(validate_username)]
ValidPassword = Annotated[str, Field(min_length=8, max_length=64), AfterValidator(validate_password)]

class UserCreate(UserBase):
    password: ValidPassword       # input : mot de passe en clair

class UserInDB(UserBase):
    hashed_password: str          # BDD : hash uniquement

class UserOut(UserBase):
    id: UUID4                     # output : PAS de password
    created_at: datetime

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "Bearer"
```

> Séparation stricte : `UserCreate` (input) → `UserInDB` (storage) → `UserOut` (output sans password).

### Hashing & Salting

```python
# services/auth.py
from passlib.context import CryptContext

class PasswordService:
    pwd_context = CryptContext(schemes=["bcrypt"])

    async def verify_password(self, password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(password, hashed_password)

    async def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)
```

| Attaque | Sans salt | Avec salt |
|---------|-----------|-----------|
| **Rainbow tables** | ✅ Vulnérable | ❌ Protégé |
| **Password spraying** | ✅ Vulnérable | ✅ Vulnérable |
| **Credential stuffing** | ✅ Vulnérable | ✅ Vulnérable |

> Le salting protège contre les rainbow tables. Contre spraying/stuffing → rate limiting, 2FA.

### Token Service

```python
class TokenService(TokenRepository):
    secret_key = "your_secret_key"
    algorithm = "HS256"
    expires_in_minutes = 60

    async def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.now(UTC) + timedelta(minutes=self.expires_in_minutes)
        token_id = await self.create(TokenCreate(expires_at=expire))
        to_encode.update({"exp": expire, "iss": "your_service_name", "sub": token_id})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode(self, encoded_token: str) -> dict:
        try:
            return jwt.decode(encoded_token, self.secret_key, algorithms=[self.algorithm])
        except JWTError:
            raise UnauthorizedException

    async def validate(self, token_id) -> bool:
        return (token := await self.get(token_id)) is not None and token.is_active

    async def deactivate(self, token_id) -> None:
        await self.update(TokenUpdate(id=token_id, is_active=False))
```

### Auth Service (orchestre tout)

```python
class AuthService:
    def __init__(self, session):
        self.password_service = PasswordService()
        self.token_service = TokenService(session)
        self.user_service = UserService(session)

    async def register_user(self, user: UserCreate) -> User:
        if await self.user_service.get(user.username):
            raise AlreadyRegisteredException
        hashed = await self.password_service.get_password_hash(user.password)
        return await self.user_service.create(UserInDB(..., hashed_password=hashed))

    async def authenticate_user(self, form_data) -> str:
        user = await self.user_service.get_user(form_data.username)
        if not user or not await self.password_service.verify_password(
            form_data.password, user.hashed_password
        ):
            raise UnauthorizedException
        return await self.token_service.create_access_token(user._asdict())

    async def get_current_user(self, credentials) -> User:
        payload = self.token_service.decode(credentials.credentials)
        if not await self.token_service.validate(payload.get("sub")):
            raise UnauthorizedException
        user = await self.user_service.get(payload.get("username"))
        if not user: raise UnauthorizedException
        return user

    async def logout(self, credentials) -> None:
        payload = self.token_service.decode(credentials.credentials)
        await self.token_service.deactivate(payload.get("sub"))
```

### Protection au niveau du router

```python
# main.py
app.include_router(auth_router, prefix="/auth")
app.include_router(
    resource_router,
    dependencies=[AuthenticateUserDep],  # protège TOUS les endpoints
    prefix="/generate",
)
```

### Auth flows secondaires (production)

| Flow | Rôle |
|------|------|
| Email verification | Anti-spambot |
| Password reset | Révoquer tous les tokens + changer hash |
| Force logout | Révoquer tous les tokens partout |
| Disable/Delete account | Bloquer login / GDPR |
| Block successive attempts | Lockout temporaire après N échecs |
| Refresh tokens | Access court + refresh long |
| 2FA / MFA | SMS, OTP, app d'auth |

> En production, considérer : **Auth0/Okta**, **Firebase Auth**, **KeyCloak**, **Amazon Cognito**.

---

## 4. OAuth2 avec GitHub

### Authorization Code Flow (7 étapes)

```
1. User clique "Login with GitHub"
2. Redirect vers GitHub (client_id, scope, state, redirect_uri)
3. User se connecte sur GitHub
4. GitHub montre le consent screen
5. GitHub renvoie un grant code vers redirect_uri
6. Serveur échange le grant code → access token
7. Serveur utilise l'access token → ressources GitHub
```

### Implémentation

```python
# 1. Redirect vers GitHub
@router.get("/oauth/github/login")
def oauth_login(request: Request) -> RedirectResponse:
    state = secrets.token_urlsafe(16)
    redirect_uri = request.url_for("oauth_github_callback_controller")
    request.session["x-csrf-state-token"] = state
    return RedirectResponse(
        url=f"https://github.com/login/oauth/authorize"
        f"?client_id={client_id}&scope=user&state={state}&redirect_uri={redirect_uri}"
    )

# 2. Échanger grant code → access token
async def exchange_grant_with_access_token(code: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://github.com/login/oauth/access_token",
            json={"client_id": client_id, "client_secret": client_secret, "code": code},
            headers={"Accept": "application/json"},
        ) as resp:
            data = await resp.json()
    return data.get("access_token", "")

# 3. Callback avec CSRF protection
def check_csrf_state(request: Request, state: str):
    if state != request.session.get("x-csrf-state-token"):
        raise HTTPException(status_code=401)

@router.get("/oauth/github/callback", dependencies=[Depends(check_csrf_state)])
async def callback(access_token: ExchangeCodeTokenDep) -> RedirectResponse:
    response = RedirectResponse(url="http://localhost:8501")
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

# 4. Récupérer user info
async def get_user_info(credentials: HTTPBearerDep) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {credentials.credentials}"},
        ) as resp:
            return await resp.json()
```

> **SessionMiddleware** obligatoire pour le CSRF : `app.add_middleware(SessionMiddleware, secret_key="...")`

### Protection CSRF

Le `state` parameter protège contre les attaques CSRF :
- Serveur génère un state aléatoire → stocké en session serveur
- Envoyé à GitHub → GitHub le renvoie dans le callback
- Serveur compare : reçu == session ? Si non → attaque → rejeter

> ⚠️ Ne jamais exposer le token GitHub au navigateur. Créer un JWT interne lié au token GitHub.

### OAuth2 Flow Types

| Flow | Quand | Sécurité |
|------|-------|----------|
| **Authorization Code (ACF)** | Apps avec backend | ✅ Haute |
| **ACF + PKCE** | Mobile (code_challenge/code_verifier) | ✅ Très haute |
| **Implicit** | SPAs sans backend | ⚠️ Moyenne |
| **Client Credentials** | Machine-to-machine | ✅ Haute |
| **Resource Owner Password** | Legacy only | ❌ Faible |
| **Device Authorization** | Smart TV, IoT | ✅ Haute |

> Pour simplifier l'implémentation OAuth : lib **`authlib`**.

---

## 5. Authorization Models

L'authorization = une fonction : `f(actor, action, resource) → allow | deny`

| Modèle | Basé sur | Granularité | Complexité |
|--------|----------|-------------|------------|
| **RBAC** | Rôles assignés aux users | Faible | Simple |
| **ReBAC** | Relations entre entités | Moyenne | Élevée |
| **ABAC** | Attributs (users, resources, env) | Très fine | Élevée |

Hiérarchie : RBAC ⊂ ReBAC ⊂ ABAC (chaque modèle peut étendre/override le précédent).

---

## 6. RBAC — Role-Based Access Control

### Principe

Assigner des **rôles** aux utilisateurs, chaque rôle a des **permissions** prédéfinies.

```
ADMIN → accès à tout (text, image, audio, user management)
USER  → accès limité (text uniquement)
```

### Implémentation simple

```python
# dependencies/auth.py
async def is_admin(user: User = Depends(AuthService.get_current_user)) -> User:
    if user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Not allowed")
    return user

# routes/resource.py
router = APIRouter(
    dependencies=[Depends(AuthService.get_current_user)],  # auth pour tous
    prefix="/generate",
)

@router.post("/image", dependencies=[Depends(is_admin)])  # admin only
async def generate_image(): ...

@router.post("/text")  # tout utilisateur authentifié
async def generate_text(): ...
```

### RBAC avancé avec rôles multiples

```python
async def has_role(user: CurrentUserDep, roles: list[str]) -> User:
    if user.role not in roles:
        raise HTTPException(status_code=403, detail="Not allowed")
    return user

@router.post("/image",
    dependencies=[Depends(lambda user: has_role(user, ["ADMIN", "MODERATOR"]))])
async def generate_image(): ...

@router.post("/text",
    dependencies=[Depends(lambda user: has_role(user, ["EDITOR"]))])
async def generate_text(): ...
```

> ⚠️ Toujours implémenter l'authorization **côté application**, pas dans le modèle GenAI (vulnérable au prompt injection).

---

## 7. ReBAC — Relationship-Based Access Control

### Principe

Authorization basée sur les **relations** entre entités (user↔user, user↔resource).

```
User "Alice" ─member_of─→ Team "Engineering"
Team "Engineering" ─owns─→ Conversation "Private Chat"
→ Alice peut accéder au "Private Chat" via son membership
```

### Avantages vs limites

| ✅ | ❌ |
|---|---|
| Résout l'explosion de rôles RBAC | Complexe à implémenter |
| Permissions héritées (parent→enfant) | Resource-intensive |
| Reverse queries efficaces | Difficile à auditer |
| Gère les hiérarchies (teams, orgs, folders) | Moins fin qu'ABAC pour les attributs dynamiques |

### Use cases

- Partager un dossier de conversations avec une équipe (héritage de permissions)
- Accès aux modèles premium achetés par le team
- Réseaux sociaux, plateformes collaboratives

---

## 8. ABAC — Attribute-Based Access Control

### Principe

Authorization basée sur les **attributs** des users, resources et environnement.

```python
# Exemples de règles ABAC
if user.subscription == "paid":       → accès aux modèles premium
if resource.is_public == True:        → visible par tous
if upload.has_pii == True:            → bloquer l'upload
if datetime.now().hour < 9:           → accès restreint hors heures de bureau
```

### Avantages vs limites

| ✅ | ❌ |
|---|---|
| Granularité maximale | Complexe pour les hiérarchies |
| Policies dynamiques (temps, lieu, attributs) | Difficile de savoir qui a accès à quoi |
| Override RBAC et ReBAC | Complexe dans les grands systèmes |

---

## 9. Hybrid Authorization + Service externe

### Hybrid : combiner RBAC + ReBAC + ABAC

```python
def authorize(user: CurrentUserDep, resource: ResourceDep, team: TeamMembershipDep) -> bool:
    # RBAC : admin → accès total
    if user.role == "ADMIN":
        return True
    # ReBAC : membre de l'équipe → accès
    if user.id in team.members:
        return True
    # ABAC : ressource publique → accès
    if resource.is_public:
        return True
    raise HTTPException(status_code=403, detail="Access Denied")

router = APIRouter(dependencies=[Depends(authorize)], prefix="/generate")
```

### Service d'authorization externe

Pour des systèmes complexes, séparer l'authorization dans un service dédié :

```python
# authorization_api.py (service séparé)
@app.get("/authorize")
def authorize(user, resource, action) -> AuthorizationResponse:
    if user.role == "ADMIN": return AuthorizationResponse(allowed=True)
    if action in user.permissions.get(resource.id, []): return AuthorizationResponse(allowed=True)
    return AuthorizationResponse(allowed=False)

# genai_api.py (service principal)
async def enforce(data: AuthorizationData) -> bool:
    response = await authorization_client.decide(data)
    if response.allowed: return True
    raise HTTPException(status_code=403, detail="Access Denied")

router = APIRouter(dependencies=[Depends(enforce)], prefix="/generate")
```

> Pour la production : **Oso**, **Permify**, **Auth0/Okta** comme authorization providers.

---

## Récap — Structure auth complète

```
project/
├── entities.py                # User, Token (SQLAlchemy models)
├── schemas.py                 # UserCreate, UserOut, TokenOut (Pydantic)
├── exceptions.py              # UnauthorizedException, AlreadyRegisteredException
├── repositories/
│   ├── users.py               # UserRepository (CRUD)
│   └── tokens.py              # TokenRepository (CRUD)
├── services/
│   └── auth.py                # PasswordService, TokenService, AuthService
├── dependencies/
│   └── auth.py                # is_admin, has_role, authorize (guards)
├── routes/
│   ├── auth.py                # /register, /token, /logout, /oauth/github/*
│   └── resource.py            # /generate/* (protégé par auth)
└── main.py                    # include_router + dependencies
```

## Dépendances

```bash
uv add passlib python-jose bcrypt aiohttp
uv add authlib  # si OAuth simplifié
```

## Points clés à retenir

1. **Basic auth = prototypage uniquement**. JWT ou OAuth2 pour la production
2. **Hashing + salting** (bcrypt) → jamais de mot de passe en clair en BDD
3. **Séparation stricte** : `UserCreate` (input) → `UserInDB` (hash) → `UserOut` (sans password)
4. **Protection au niveau router** : `dependencies=[AuthenticateUserDep]` protège tout le groupe
5. **OAuth2 ACF** : redirect → consent → grant code → exchange → access token
6. **CSRF protection** : state parameter en session serveur, jamais en cookie
7. **Ne pas exposer le token du provider** au navigateur → créer un JWT interne
8. **RBAC** suffisant pour la plupart des apps. ReBAC pour les hiérarchies, ABAC pour le fin-grained
9. **Authorization côté application** → pas dans le modèle GenAI (prompt injection)
10. Pour les systèmes complexes → **service d'authorization séparé** ou provider (Oso, Auth0)