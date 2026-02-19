# Chapter 7 — Integrating Databases into AI Services

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Quand utiliser (ou non) une BDD](#1-quand-utiliser-ou-non-une-bdd)
2. [Taxonomie des bases de données](#2-taxonomie-des-bases-de-données)
3. [Setup : PostgreSQL + SQLAlchemy + Alembic](#3-setup--postgresql--sqlalchemy--alembic)
4. [ORM Models avec SQLAlchemy](#4-orm-models-avec-sqlalchemy)
5. [Engine, sessions et dependency injection](#5-engine-sessions-et-dependency-injection)
6. [CRUD endpoints](#6-crud-endpoints)
7. [Repository & Service patterns](#7-repository--service-patterns)
8. [Migrations avec Alembic](#8-migrations-avec-alembic)
9. [Stocker les données en streaming](#9-stocker-les-données-en-streaming)

---

## 1. Quand utiliser (ou non) une BDD

### Cas où on peut s'en passer

1. L'app redémarre à zéro à chaque session
2. Recalculer les données est simple/pas cher
3. Les données tiennent en mémoire
4. L'app tolère les pertes de données
5. Pas de partage entre sessions/instances
6. Les données viennent de systèmes externes
7. L'utilisateur accepte d'attendre un recalcul
8. Le stockage fichier/browser/cloud suffit
9. C'est un PoC

### Quand on a besoin d'une BDD

Dès qu'on doit : persister l'état, stocker des données utilisateur, retrouver/manipuler/analyser efficacement. Features clés : backup/restore, accès concurrent, indexation, caching, RBAC.

---

## 2. Taxonomie des bases de données

### Structure mentale commune

```
Server (CPU, RAM, Storage)
└── Database(s)
    └── Schema(s)
        └── Table(s) / Collection(s)
            └── Row(s) / Document(s)
```

### Types de BDD

| Type | Modèle | Exemples | Use cases |
|------|--------|----------|-----------|
| **Relationnel (SQL)** | Tables, lignes, colonnes | PostgreSQL, MySQL, SQLite | Données structurées, CRUD, transactions ACID |
| **Key-value** | Paires clé-valeur | Redis, DynamoDB | Caching, sessions |
| **Graph** | Nœuds et arêtes | Neo4j, ArangoDB | Réseaux sociaux, recommandations, RAG graph |
| **Document** | Documents JSON-like | MongoDB, CouchDB | CMS, schémas flexibles |
| **Vector** | Vecteurs haute dimension | Qdrant, Pinecone, Weaviate | RAG, recherche sémantique |
| **Wide-column** | Tables flexibles | Cassandra, HBase | Time-series, logging |

### Architecture multi-BDD d'un service RAG en prod

```
[LLM Service RAG-enabled]
    ├── PostgreSQL      → utilisateurs, conversations, monitoring
    ├── Neo4j/ArangoDB  → relations entre documents (RAG graph)
    ├── Qdrant          → embeddings (recherche sémantique)
    ├── Redis           → cache des réponses LLM fréquentes
    └── MongoDB         → templates de prompts versionnés
```

---

## 3. Setup : PostgreSQL + SQLAlchemy + Alembic

### Lancer Postgres avec Docker

```bash
docker run -p 5432:5432 \
    -e POSTGRES_USER=fastapi \
    -e POSTGRES_PASSWORD=mysecretpassword \
    -e POSTGRES_DB=backend_db \
    -e PGDATA=/var/lib/postgresql/data \
    -v "$(pwd)"/dbstorage:/var/lib/postgresql/data \
    postgres:latest
```

### Dépendances

```bash
uv add alembic sqlalchemy psycopg3
```

| Package | Rôle |
|---------|------|
| **psycopg3** | Driver PostgreSQL pour Python |
| **SQLAlchemy** | ORM + toolkit SQL |
| **Alembic** | Migrations de schéma (version control BDD) |

### ORM — Object Relational Mapper

| BDD | Python (ORM) |
|-----|-------------|
| Table | Classe |
| Colonne | Attribut de classe |
| Ligne | Instance de classe |

```python
# SQL brut
SELECT * FROM users WHERE id = 1;

# SQLAlchemy ORM
session.query(User).filter(User.id == 1).first()
```

---

## 4. ORM Models avec SQLAlchemy

```python
# entities.py
from datetime import UTC, datetime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column()
    model_type: Mapped[str] = mapped_column(index=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    prompt_content: Mapped[str] = mapped_column()
    response_content: Mapped[str] = mapped_column()
    prompt_tokens: Mapped[int | None] = mapped_column()
    response_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    is_success: Mapped[bool | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
```

### Points clés

- `Mapped[int]` → colonne NOT NULL. `Mapped[int | None]` → nullable
- `mapped_column()` déduit le type SQL depuis le type hint
- `index=True` sur `model_type` et `conversation_id` → requêtes de filtrage plus rapides
- `cascade="all, delete-orphan"` → supprime les messages si la conversation est supprimée
- `relationship()` + `back_populates` → navigation bidirectionnelle entre models
- `onupdate=datetime.now(UTC)` → mise à jour automatique de `updated_at`

---

## 5. Engine, sessions et dependency injection

### Database engine (async)

```python
# database.py
from sqlalchemy.ext.asyncio import create_async_engine
from entities import Base

database_url = "postgresql+psycopg://fastapi:mysecretpassword@localhost:5432/backend_db"
engine = create_async_engine(database_url, echo=True)

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)    # ⚠️ dev only
        await conn.run_sync(Base.metadata.create_all)
```

> `create_all()` ne peut que **créer** des tables, pas les modifier. Pour la prod → Alembic.

### Lifespan FastAPI

```python
# main.py
@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()       # startup
    yield
    await engine.dispose()  # shutdown — libère les connexions

app = FastAPI(lifespan=lifespan)
```

### Session factory + dependency injection

```python
# database.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession,
    autocommit=False,   # contrôle manuel des transactions
    autoflush=False,     # pas de flush automatique → évite les updates non voulus
)

async def get_db_session():
    try:
        async with async_session() as session:
            yield session
    except:
        await session.rollback()   # rollback en cas d'erreur
        raise
    finally:
        await session.close()      # toujours fermer la session

# Type alias réutilisable
DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
```

Points clés :
- `autocommit=False` + `autoflush=False` → contrôle total des transactions
- `yield session` → FastAPI gère le lifecycle via le context manager
- `rollback()` en cas d'exception → pas de données corrompues
- `DBSessionDep` → type alias injectable dans n'importe quel handler

---

## 6. CRUD endpoints

### Schemas Pydantic (découplés des models SQLAlchemy)

```python
# schemas.py
from pydantic import BaseModel, ConfigDict

class ConversationBase(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # lit les attributs SQLAlchemy
    title: str
    model_type: str

class ConversationCreate(ConversationBase): pass
class ConversationUpdate(ConversationBase): pass

class ConversationOut(ConversationBase):
    id: int
    created_at: datetime
    updated_at: datetime
```

> `from_attributes=True` → Pydantic peut lire directement les attributs d'un objet SQLAlchemy.

> Séparer Pydantic et SQLAlchemy = duplication de code mais découplage API/BDD. Alternative : `sqlmodel` (mais limité pour les cas complexes).

### Dependency pour vérifier l'existence d'un record

```python
async def get_conversation(conversation_id: int, session: DBSessionDep) -> Conversation:
    async with session.begin():
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

GetConversationDep = Annotated[Conversation, Depends(get_conversation)]
```

Réutilisé dans GET, PUT, DELETE → évite de dupliquer la vérification d'existence.

### Les 5 endpoints CRUD

```python
# GET /conversations — liste avec pagination
@app.get("/conversations")
async def list_conversations(session: DBSessionDep, skip: int = 0, take: int = 100):
    result = await session.execute(select(Conversation).offset(skip).limit(take))
    return [ConversationOut.model_validate(c) for c in result.scalars().all()]

# GET /conversations/{id} — détail
@app.get("/conversations/{id}")
async def get_conversation(conversation: GetConversationDep):
    return ConversationOut.model_validate(conversation)

# POST /conversations — création (201)
@app.post("/conversations", status_code=201)
async def create_conversation(conversation: ConversationCreate, session: DBSessionDep):
    new = Conversation(**conversation.model_dump())
    session.add(new)
    await session.commit()
    await session.refresh(new)
    return ConversationOut.model_validate(new)

# PUT /conversations/{id} — mise à jour (202)
@app.put("/conversations/{id}", status_code=202)
async def update_conversation(updated: ConversationUpdate,
                               conversation: GetConversationDep, session: DBSessionDep):
    for key, value in updated.model_dump().items():
        setattr(conversation, key, value)
    await session.commit()
    await session.refresh(conversation)
    return ConversationOut.model_validate(conversation)

# DELETE /conversations/{id} — suppression (204)
@app.delete("/conversations/{id}", status_code=204)
async def delete_conversation(conversation: GetConversationDep, session: DBSessionDep):
    await session.delete(conversation)
    await session.commit()
```

### Status codes par opération

| Opération | Status code |
|-----------|-------------|
| GET (list/detail) | 200 |
| POST (create) | 201 Created |
| PUT (update) | 202 Accepted |
| DELETE | 204 No Content |

---

## 7. Repository & Service patterns

### Architecture Onion

```
Controllers (routers) → Services (business logic) → Repositories (data access) → Database
```

### Repository : interface abstraite

```python
# repositories/interfaces.py
from abc import ABC, abstractmethod

class Repository(ABC):
    @abstractmethod
    async def list(self) -> list[Any]: pass

    @abstractmethod
    async def get(self, uid: int) -> Any: pass

    @abstractmethod
    async def create(self, record: Any) -> Any: pass

    @abstractmethod
    async def update(self, uid: int, record: Any) -> Any: pass

    @abstractmethod
    async def delete(self, uid: int) -> None: pass
```

### Repository : implémentation concrète

```python
# repositories/conversations.py
class ConversationRepository(Repository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, skip: int, take: int) -> list[Conversation]:
        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).offset(skip).limit(take)
            )
            return [r for r in result.scalars().all()]

    async def get(self, conversation_id: int) -> Conversation | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            return result.scalars().first()

    async def create(self, conversation: ConversationCreate) -> Conversation:
        new = Conversation(**conversation.model_dump())
        async with self.session.begin():
            self.session.add(new)
            await self.session.commit()
            await self.session.refresh(new)
            return new

    async def update(self, conversation_id, updated) -> Conversation | None:
        conversation = await self.get(conversation_id)
        if not conversation: return None
        for key, value in updated.model_dump().items():
            setattr(conversation, key, value)
        await self.session.commit()
        await self.session.refresh(conversation)
        return conversation

    async def delete(self, conversation_id: int) -> None:
        conversation = await self.get(conversation_id)
        if not conversation: return
        await self.session.delete(conversation)
        await self.session.commit()
```

### Service : étend le repository avec la business logic

```python
# services/conversations.py
class ConversationService(ConversationRepository):
    async def list_messages(self, conversation_id: int) -> list[Message]:
        result = await self.session.execute(
            select(Message).where(Message.conversation_id == conversation_id)
        )
        return [m for m in result.scalars().all()]
```

### Controllers refactorisés (propres)

```python
# routers/conversations.py
router = APIRouter(prefix="/conversations")

@router.get("")
async def list_conversations(session: DBSessionDep, skip: int = 0, take: int = 100):
    conversations = await ConversationRepository(session).list(skip, take)
    return [ConversationOut.model_validate(c) for c in conversations]

@router.get("/{conversation_id}/messages")
async def list_messages(conversation: GetConversationDep, session: DBSessionDep):
    messages = await ConversationService(session).list_messages(conversation.id)
    return [MessageOut.model_validate(m) for m in messages]

# main.py
app.include_router(conversations_router)
```

### Bonnes pratiques repository/service

- Ne pas mettre de business logic dans les repositories (data access only)
- Ne pas coupler les services à une implémentation spécifique de repository
- Gérer les transactions et exceptions proprement
- Attention aux performances (JOINs complexes)
- Nommage cohérent des méthodes et classes

---

## 8. Migrations avec Alembic

### Alembic = Git pour les schémas de BDD

### Setup

```bash
alembic init alembic
```

Structure générée :

```
project/
├── alembic.ini
└── alembic/
    ├── env.py              # config : connexion BDD + metadata SQLAlchemy
    ├── script.py.mako
    └── versions/           # fichiers de migration
```

### Connecter Alembic aux models SQLAlchemy

```python
# alembic/env.py
from entities import Base
from settings import AppSettings

settings = AppSettings()
target_metadata = Base
db_url = str(settings.pg_dsn)
```

### Workflow de migration

```bash
# 1. Générer une migration (compare models vs BDD actuelle)
alembic revision --autogenerate -m "Initial Migration"

# 2. Appliquer la migration
alembic upgrade head

# 3. Reverter si besoin
alembic downgrade -1
```

### Fichier de migration généré

```python
# alembic/versions/24c35f32b152.py
revision = "24c35f32b152"
down_revision = None

def upgrade():
    op.create_table("conversations",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("model_type", sa.String, index=True, nullable=False),
        # ... autres colonnes
    )
    op.create_table("messages", ...)

def downgrade():
    op.drop_table("messages")
    op.drop_table("conversations")
```

### Règles importantes

- **Toujours commit** le fichier de migration dans Git après l'avoir appliqué
- **Ne jamais rééditer** un fichier de migration déjà appliqué (Alembic ne détectera pas les changements)
- Pour modifier le schéma → créer une **nouvelle migration**
- Si drift entre BDD et historique → supprimer les fichiers `versions/`, truncate `alembic_revision`, réinitialiser
- Alembic maintient une table `alembic_versions` dans la BDD pour tracker les migrations appliquées

---

## 9. Stocker les données en streaming

### Le problème

On ne peut pas écrire en BDD relationnelle pendant un stream (ACID compliance difficile). Solution : **BackgroundTask** après le stream.

### Pattern : tee + background task

```python
from itertools import tee

async def store_message(prompt, response_content, conversation_id, session):
    message = Message(
        conversation_id=conversation_id,
        prompt_content=prompt,
        response_content=response_content,
    )
    await MessageRepository(session).create(message)

@app.get("/text/generate/stream")
async def stream_llm(prompt: str, background_task: BackgroundTasks,
                     session: DBSessionDep, conversation=Depends(get_conversation)):
    # ... invoke LLM, get response_stream
    stream_1, stream_2 = tee(response_stream)  # dupliquer le stream

    background_task.add_task(
        store_message, prompt, "".join(stream_1), conversation.id, session
    )
    return StreamingResponse(stream_2)
```

**Comment ça marche** :
1. `tee()` crée deux copies du stream
2. `stream_2` → envoyé au client via `StreamingResponse`
3. `stream_1` → concaténé en string et stocké en BDD via background task
4. Le stockage se fait **après** que le stream est terminé

### Générer un titre de conversation automatiquement

```python
async def create_conversation(initial_prompt: str, session: AsyncSession):
    completion = await async_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Suggest a title based on the user prompt"},
            {"role": "user", "content": initial_prompt},
        ],
        model="gpt-3.5-turbo",
    )
    title = completion.choices[0].message.content
    conversation = Conversation(title=title, ...)
    return await ConversationRepository(session).create(conversation)
```

---

## Récap — Structure du projet

```
project/
├── main.py                  # app FastAPI + lifespan + include_router
├── database.py              # engine, session factory, DBSessionDep
├── entities.py              # SQLAlchemy ORM models (Conversation, Message)
├── schemas.py               # Pydantic schemas (Create, Update, Out)
├── settings.py              # AppSettings (Pydantic Settings)
├── repositories/
│   ├── interfaces.py        # Repository ABC
│   ├── conversations.py     # ConversationRepository
│   └── messages.py          # MessageRepository
├── services/
│   └── conversations.py     # ConversationService (extends Repository)
├── routers/
│   └── conversations.py     # APIRouter + controllers
└── alembic/
    ├── env.py               # config Alembic → SQLAlchemy metadata
    └── versions/             # fichiers de migration
```

## Dépendances

```bash
uv add sqlalchemy alembic psycopg3
```

## Points clés à retenir

1. **SQLAlchemy async** : `create_async_engine` + `async_sessionmaker` + `AsyncSession`
2. **Session = dependency injectable** via `DBSessionDep` → réutilisable partout
3. **Pydantic séparé de SQLAlchemy** : `from_attributes=True` + `model_validate()` pour convertir
4. **Repository pattern** : isole le data access → controllers propres
5. **Service pattern** : étend le repository avec la business logic (queries complexes)
6. **Alembic** : `revision --autogenerate` → `upgrade head` → commit le fichier
7. **Streaming + BDD** : `tee()` pour dupliquer le stream + `BackgroundTask` pour stocker après
8. **Ne jamais hard-coder** les credentials → Pydantic Settings + `.env`
9. **`autocommit=False` + `autoflush=False`** → contrôle total des transactions
10. **Toujours `rollback()` en cas d'exception** et `close()` dans le `finally`