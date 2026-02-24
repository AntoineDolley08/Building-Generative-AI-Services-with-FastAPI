# Building Generative AI Services with FastAPI

> Cahier de notes personnel du livre *Building Generative AI Services with FastAPI* d'**Ali Parandeh** (O'Reilly, 2025).

Ce repo contient une **fiche de notes par chapitre** avec les concepts cles, exemples de code, et patterns d'architecture. Les chapitres qui incluent un projet ont le code source complet dans un sous-dossier dedie.

---

## Structure du repo

```
.
├── Part 1. Developing AI Services
│   ├── Chapter 1.  Introduction
│   ├── Chapter 2.  Getting Started with FastAPI
│   ├── Chapter 3.  AI Integration and Model Serving
│   └── Chapter 4.  Implementing Type-Safe AI Services
│
├── Part 2. Communicating with External Systems
│   ├── Chapter 5.  Achieving Concurrency in AI Workloads
│   ├── Chapter 6.  Real-Time Communication with Generative Models
│   └── Chapter 7.  Integrating Databases into AI Services
│
└── Part 3. Securing, Optimizing, Testing, and Deploying AI Services
    ├── Chapter 8.  Authentication & Authorization
    ├── Chapter 9.  Securing AI Services
    ├── Chapter 10. Optimizing AI Services
    ├── Chapter 11. Testing AI Services
    └── Chapter 12. Deployment of AI Services
```

---

## Table des chapitres

### Part 1 — Developing AI Services

| # | Chapitre | Concepts cles | Fiche | Code |
|---|----------|---------------|-------|------|
| 1 | Why Generative AI Services Will Power Future Applications | Passage rule-based vers model-based, capacites GenAI (facilitation creative, personnalisation, interfaces langage naturel) | [Fiche](Part%201.%20Develloping%20AI%20services/Chapter%201.%20Introduction.md) | — |
| 2 | Getting Started with FastAPI | Architecture FastAPI, structures de projet (flat, nested, modulaire), design Onion/couches, boucle ASGI, dependency injection, tooling Python (Ruff, mypy) | [Fiche](Part%201.%20Develloping%20AI%20services/Chapter%202.%20Getting%20Started%20with%20FastAPI/Getting%20started%20with%20FastAPI.md) | [example-genai-fastapi-service/](Part%201.%20Develloping%20AI%20services/Chapter%202.%20Getting%20Started%20with%20FastAPI/example-genai-fastapi-service/) |
| | *Tuto : Setup FastAPI + uv* | | [Tuto](Part%201.%20Develloping%20AI%20services/Chapter%202.%20Getting%20Started%20with%20FastAPI/Tuto%20FastAPI%20uv%20setup.md) | |
| 3 | AI Integration and Model Serving | Architecture Transformer, tokenization et embeddings, LLM (GPT, LLaMA), modeles de diffusion, strategies de serving (lifespan, serveurs externes, API providers), middleware monitoring | [Fiche](Part%201.%20Develloping%20AI%20services/Chapter%203.%20Ai%20Integration%20and%20Model%20Serving/Serving%20language%20models.md) | [all-models-app/](Part%201.%20Develloping%20AI%20services/Chapter%203.%20Ai%20Integration%20and%20Model%20Serving/all-models-app/) |
| 4 | Implementing Type-Safe AI Services | Type annotations et TypeAlias, modeles Pydantic, contraintes et validateurs, computed fields, Pydantic Settings, serialisation, integration FastAPI | [Fiche](Part%201.%20Develloping%20AI%20services/Chapter%204.%20Implementing%20Type-Safe%20AI%20Services/Implementing%20Type-Safe%20AI%20Services.md) | — |

### Part 2 — Communicating with External Systems

| # | Chapitre | Concepts cles | Fiche | Code |
|---|----------|---------------|-------|------|
| 5 | Achieving Concurrency in AI Workloads | Concurrence vs parallelisme, async/await, event loop, thread pool, GIL, multiprocessing, web scraper async + RAG, background tasks | [Fiche](Part%202.%20Communicating%20with%20External%20Systems/Chapter%205.%20Achieving%20Concurrency%20in%20AI%20Workloads/Achieving%20Concurrency%20in%20AI%20Workloads.md) | [async_vs_sync/](Part%202.%20Communicating%20with%20External%20Systems/Chapter%205.%20Achieving%20Concurrency%20in%20AI%20Workloads/async_vs_sync/) &#124; [talk_to_documents/](Part%202.%20Communicating%20with%20External%20Systems/Chapter%205.%20Achieving%20Concurrency%20in%20AI%20Workloads/talk_to_documents/) &#124; [talk_to_the_web/](Part%202.%20Communicating%20with%20External%20Systems/Chapter%205.%20Achieving%20Concurrency%20in%20AI%20Workloads/talk_to_the_web/) |
| 6 | Real-Time Communication with Generative Models | SSE (Server-Sent Events), WebSocket bidirectionnel, streaming responses, connection managers, retry client, CORS middleware | [Fiche](Part%202.%20Communicating%20with%20External%20Systems/Chapter%206.%20Real-Time%20Communication%20with%20Generative%20Models/Real-Time%20Communication%20with%20Generative%20Models.md) | — |
| 7 | Integrating Databases into AI Services | SQLAlchemy ORM async, migrations Alembic, CRUD endpoints, pattern Repository/Service, schemas Pydantic, stockage streaming + background tasks | [Fiche](Part%202.%20Communicating%20with%20External%20Systems/Chapter%207.%20Integrating%20Databases%20into%20AI%20Services/Integrating%20Databases%20into%20AI%20Services.md) | [database-ai-services/](Part%202.%20Communicating%20with%20External%20Systems/Chapter%207.%20Integrating%20Databases%20into%20AI%20Services/database-ai-services/) |

### Part 3 — Securing, Optimizing, Testing, and Deploying AI Services

| # | Chapitre | Concepts cles | Fiche | Code |
|---|----------|---------------|-------|------|
| 8 | Authentication & Authorization | Methodes auth (Basic, JWT, OAuth2), hashing bcrypt, gestion de tokens, modeles d'autorisation (RBAC, ReBAC, ABAC), OAuth2 GitHub | [Fiche](Part%203.%20Securing%2C%20Optimizing%2C%20Testing%2C%20and%20Deploying%20AI%20Services/Chapter%208.%20Authentication%20and%20Authorization/Authentication%20and%20Authorization.md) | [authentication-authorization/](Part%203.%20Securing%2C%20Optimizing%2C%20Testing%2C%20and%20Deploying%20AI%20Services/Chapter%208.%20Authentication%20and%20Authorization/authentication-authorization/) |
| 9 | Securing AI Services | Menaces securite GenAI, guardrails input/output, filtrage topical, evaluation moderation, rate limiting (token bucket, leaky bucket, fixed/sliding window), throttling streams | [Fiche](Part%203.%20Securing%2C%20Optimizing%2C%20Testing%2C%20and%20Deploying%20AI%20Services/Chapter%209.%20Securing%20AI%20Services/Securing%20AI%20Services.md) | — |
| 10 | Optimizing AI Services | Batch processing, caching (keyword, semantique, contexte), quantization (FP16, INT8, INT4), structured outputs JSON, prompt engineering (RCT, few-shot, chain-of-thought), fine-tuning | [Fiche](Part%203.%20Securing%2C%20Optimizing%2C%20Testing%2C%20and%20Deploying%20AI%20Services/Chapter%2010.%20Optimizing%20AI%20Services/Optimizing%20AI%20Services.md) | — |
| 11 | Testing AI Services | Strategies de test (pyramide, trophee, honeycomb), pytest, fixtures et parametrisation, test doubles (Fake, Dummy, Stub, Spy, Mock), tests d'integration, tests comportementaux GenAI | [Fiche](Part%203.%20Securing%2C%20Optimizing%2C%20Testing%2C%20and%20Deploying%20AI%20Services/Chapter%2011.%20Testing%20AI%20Services/Testing%20AI%20Services.md) | [testing-ai-services/](Part%203.%20Securing%2C%20Optimizing%2C%20Testing%2C%20and%20Deploying%20AI%20Services/Chapter%2011.%20Testing%20AI%20Services/testing-ai-services/) |
| 12 | Deployment of AI Services | Options deploiement (VM, serverless, managed, conteneurs), Docker, multi-stage builds, Docker Compose, networking et storage, support GPU, securite (non-root) | [Fiche](Part%203.%20Securing%2C%20Optimizing%2C%20Testing%2C%20and%20Deploying%20AI%20Services/Chapter%2012.%20Deployment%20of%20AI%20Services/Deployment%20of%20AI%20Services.md) | — |

---

## Sujets couverts

| Domaine | Technologies / Concepts |
|---------|------------------------|
| **Framework** | FastAPI, Pydantic, Uvicorn, uv |
| **LLM Providers** | OpenAI API, Anthropic API, Google Gemini API |
| **Model Serving** | LangChain, BentoML, Hugging Face Transformers |
| **Base de donnees** | PostgreSQL, SQLAlchemy (async), Alembic |
| **Vector Store** | Qdrant |
| **Authentification** | JWT (python-jose), OAuth2, bcrypt, passlib |
| **Securite** | CORS, rate limiting, input validation, OWASP |
| **Caching** | Redis (fastapi-cache), semantic caching (gptcache) |
| **Concurrence** | asyncio, SSE, WebSocket, streaming |
| **Tests** | pytest, HTTPX AsyncClient, mocking |
| **Deploiement** | Docker, Docker Compose, CI/CD |
| **Optimisation** | Batch processing, model quantization (GPTQ), prompt engineering |

---

## Comment utiliser ce repo

Chaque chapitre a sa propre **fiche de notes** en markdown. Pour relire un sujet :

1. Consulter la **table des chapitres** ci-dessus
2. Cliquer sur le lien **Fiche** pour lire les notes du chapitre
3. Si le chapitre a du code, ouvrir le dossier **Code** pour voir l'implementation

Les projets code utilisent **uv** comme gestionnaire de paquets. Pour lancer un projet :

```bash
cd "chemin/vers/le/projet"
uv sync
uv run uvicorn main:app --reload
```

---

## Reference

**Livre** : *Building Generative AI Services with FastAPI* — Ali Parandeh (O'Reilly, 2025)
