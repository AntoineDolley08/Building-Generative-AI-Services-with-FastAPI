# Chapter 9 — Securing AI Services

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Menaces et abus des services GenAI](#1-menaces-et-abus-des-services-genai)
2. [Guardrails — Input](#2-guardrails--input)
3. [Guardrails — Output](#3-guardrails--output)
4. [Exécution async des guardrails](#4-exécution-async-des-guardrails)
5. [Rate Limiting](#5-rate-limiting)
6. [Throttling des streams](#6-throttling-des-streams)

---

## 1. Menaces et abus des services GenAI

### Intents malveillants

| Intent | Exemples |
|--------|----------|
| **Dishonesty** | Plagiat, triche, falsification de documents |
| **Propaganda** | Impersonation, extrémisme, campagnes d'influence |
| **Deception** | Faux avis, phishing, profils synthétiques, deepfakes |

### OWASP Top 10 LLM Vulnerabilities

| Risque | Description |
|--------|-------------|
| **Prompt injection** | Manipuler les inputs pour contrôler les réponses |
| **Insecure output handling** | Outputs non sanitisés → exécution de code |
| **Training data poisoning** | Données corrompues injectées dans les sources d'entraînement |
| **Model DoS** | Surcharge par payloads lourds et requêtes concurrentes |
| **Supply chain vulnerabilities** | Composants tiers compromis |
| **Sensitive info leakage** | Exposition de données privées |
| **Insecure plugin design** | Vulnérabilités dans les intégrations tierces |
| **Excessive agency** | Trop d'autonomie pour le LLM → actions non voulues |
| **Overreliance** | Dépendance excessive aux outputs LLM |
| **Model theft** | Copie/usage non autorisé du modèle |

---

## 2. Guardrails — Input

### Types de guardrails en entrée

| Guardrail | Rôle | Exemple |
|-----------|------|---------|
| **Topical** | Bloquer les sujets hors-topic/sensibles | Empêcher les discussions politiques |
| **Direct prompt injection** | Empêcher le jailbreaking et la révélation de system prompts | Bloquer "ignore previous instructions" |
| **Indirect prompt injection** | Bloquer le contenu malveillant via fichiers/URLs externes | Sanitiser les payloads cachés dans les images/documents |
| **Moderation** | Conformité légale, marque, guidelines | Flaguer profanité, PII, contenu explicite |
| **Attribute** | Valider les propriétés de l'input | Longueur, taille fichier, format, structure |

### Implémentation : guardrail topical par LLM auto-évaluation

```python
# System prompt pour l'auto-évaluateur
guardrail_system_prompt = """
Your role is to assess user queries as valid or invalid.
Allowed topics: API Development, FastAPI, Building GenAI systems.
If allowed, say 'allowed' otherwise say 'disallowed'.
"""

class LLMClient:
    def __init__(self, system_prompt: str):
        self.client = AsyncOpenAI()
        self.system_prompt = system_prompt

    async def invoke(self, user_query: str) -> str | None:
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0,  # déterministe pour la classification
        )
        return response.choices[0].message.content

# Validation Pydantic de la réponse
ClassificationResponse = Annotated[
    str | None, AfterValidator(check_classification_response)
]

class TopicalGuardResponse(BaseModel):
    classification: ClassificationResponse

async def is_topic_allowed(user_query: str) -> TopicalGuardResponse:
    response = await LLMClient(guardrail_system_prompt).invoke(user_query)
    return TopicalGuardResponse(classification=response)
```

Points clés :
- `temperature=0` → réponse déterministe pour la classification
- Validation Pydantic de la réponse du guardrail (failsafe si le LLM retourne n'importe quoi)
- Même technique applicable pour : détection PII, profanité, prompt injection

---

## 3. Guardrails — Output

### Types de guardrails en sortie

| Guardrail | Rôle | Métriques |
|-----------|------|-----------|
| **Hallucination/fact-checking** | Bloquer les réponses hallucinées | Relevancy, coherence, consistency, fluency |
| **Moderation** | Conformité marque/corporate | Readability, toxicity, sentiment, mentions concurrents |
| **Syntax checks** | Valider la structure des outputs | JSON schema, paramètres de fonctions, sélection d'outils |

### Implémentation : moderation par G-Eval

G-Eval = framework d'évaluation par LLM avec 4 composants : domain, criteria, steps, score (1-5).

```python
# System prompt G-Eval
f"""
You are a moderation assistant.
Your role is to detect content about {domain} and mark severity.

## Criteria
{criteria}

## Instructions
{steps}

## Evaluation (score only!)
"""

# Moderation guardrail
class ModerationResponse(BaseModel):
    score: Annotated[int, Field(ge=1, le=5)]

async def g_eval_moderate_content(chat_response: str, threshold: int = 3) -> bool:
    response = await LLMClient(guardrail_system_prompt).invoke(chat_response)
    g_eval_score = ModerationResponse(score=response).score
    return g_eval_score >= threshold  # True = flagué
```

### Thresholds — trade-offs

| Trop de false positives | Trop de false negatives |
|---|---|
| Users frustrés, UX dégradée | Réputation endommagée, abus du système |
| Requêtes valides bloquées | Coûts qui explosent |

→ Évaluer les risques et trouver le bon équilibre pour ton use case.

---

## 4. Exécution async des guardrails

### Pattern : guardrails en parallèle avec la génération

```python
async def invoke_llm_with_guardrails(user_query: str) -> str:
    # Lancer les deux tâches en parallèle
    topical_guardrail_task = asyncio.create_task(is_topic_allowed(user_query))
    chat_task = asyncio.create_task(llm_client.invoke(user_query))

    while True:
        done, _ = await asyncio.wait(
            [topical_guardrail_task, chat_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if topical_guardrail_task in done:
            if not topical_guardrail_task.result():
                chat_task.cancel()  # ← annuler la génération si guardrail triggered
                return "Sorry, I can only talk about building GenAI services"

        elif chat_task in done:
            chat_response = chat_task.result()
            # Output guardrail (séquentiel après la génération)
            if not await g_eval_moderate_content(chat_response):
                return "Sorry, we can't recommend specific tools at this time"
            return chat_response

        else:
            await asyncio.sleep(0.1)

# Injection dans le controller
@router.post("/text/generate")
async def generate_text(response: Annotated[str, Depends(invoke_llm_with_guardrails)]):
    return response
```

### Architecture du flow

```
User query
    ├──→ [Input guardrail (topical)] ──→ si triggered → cancel chat → return canned response
    └──→ [LLM generation] ──→ response
                                   ↓
                          [Output guardrail (moderation)]
                                   ↓
                          si flagué → return canned response
                          sinon → return response
```

### Bonnes pratiques guardrails

- **Fast failure** : exit dès qu'un guardrail est triggered
- **Sélectivité** : n'utiliser que les guardrails pertinents pour ton use case
- **Async** : exécuter les guardrails en parallèle, pas séquentiellement
- **Sampling** : appliquer les guardrails lents sur un échantillon de requêtes sous forte charge
- **Combiner** : LLM auto-eval + règles classiques + ML traditionnel pour réduire les faux positifs/négatifs
- **Dernière ligne** : ne considérer que le dernier message pour éviter la confusion sur les longs contextes

### Frameworks disponibles

| Framework | Type |
|-----------|------|
| **NVIDIA NeMo Guardrails** | Open source |
| **LLM-Guard** | Open source |
| **Guardrails AI** | Open source |
| **OpenAI Moderation API** | Commercial |
| **Azure AI Content Safety** | Commercial |
| **Google Guardrails API** | Commercial |

> Les guardrails restent probabilistes. Des attaques avancées peuvent les contourner → défense en profondeur.

---

## 5. Rate Limiting

### Rate Limiting vs Throttling

| Concept | Rôle |
|---------|------|
| **Rate Limiting** | Contrôler le **nombre** de requêtes par période (rejeter si dépassé) |
| **Throttling** | Contrôler le **débit** de traitement (ralentir, pas rejeter) |

### 4 stratégies de rate limiting

| Stratégie | Principe | Complexité | Use cases |
|-----------|----------|------------|-----------|
| **Token Bucket** | Bucket rempli à taux constant, chaque requête consomme un token. Burst autorisé. | Élevée | APIs générales, systèmes event-driven |
| **Leaky Bucket** | Queue traitée à taux constant. Overflow → rejet. | Simple | Services nécessitant des temps de réponse constants |
| **Fixed Window** | Limite par fenêtre fixe (ex: 100/min) | Simple | Tiers gratuits, batch processing, policies strictes |
| **Sliding Window** | Comptage sur fenêtre glissante | Moyenne | Chat conversationnel, tiers premium, burst handling |

### Implémentation avec slowapi

```bash
uv add slowapi
```

```python
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# Rate limiter global (par IP)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "60 per hour", "2/5seconds"],
)
app.state.limiter = limiter

# Exception handler custom → retourne retry_after
@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    retry_after = int(exc.description.split(" ")[-1])
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded", "retry_after_seconds": retry_after},
        headers={"Retry-After": str(retry_after)},
    )

app.add_middleware(SlowAPIMiddleware)
```

### Rate limits par endpoint

```python
@app.post("/generate/text")
@limiter.limit("5/minute")               # 5 requêtes/min pour text
def generate_text(request: Request, ...): ...

@app.post("/generate/image")
@limiter.limit("1/minute")               # 1 requête/min pour image (plus coûteux)
def generate_image(request: Request, ...): ...

@app.get("/health")
@limiter.exempt                           # exempté (healthcheck externe)
def health(request: Request): ...
```

> ⚠️ Le décorateur `@limiter.limit` doit être **après** `@app.post`. Et le handler **doit** recevoir `request: Request`.

### Rate limits par utilisateur (pas seulement IP)

```python
@app.post("/generate/text")
@limiter.limit("10/minute", key_func=get_current_user)  # par user ID
def generate_text(request: Request): ...
```

> Les users peuvent contourner le rate limiting par IP avec VPN/proxies → le user-based est plus robuste.

### Rate limits multi-instances avec Redis

```bash
uv add coredis
docker run --name redis-cache -d -p 6379:6379 redis
```

```python
app.state.limiter = Limiter(storage_uri="redis://localhost:6379")
```

Sans Redis, chaque instance a son propre compteur → les limites ne sont pas respectées derrière un load balancer.

### Rate limiting WebSocket

```python
from fastapi_limiter.depends import WebSocketRateLimiter

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, user_id=Depends(get_current_user)):
    ratelimit = WebSocketRateLimiter(times=1, seconds=5)
    await ws_manager.connect(websocket)
    try:
        while True:
            prompt = await ws_manager.receive(websocket)
            await ratelimit(websocket, context_key=user_id)
            async for chunk in chat_client.chat_stream(prompt, "ws"):
                await ws_manager.send(chunk, websocket)
    except WebSocketRateLimitException:
        await websocket.send_text("Rate limit exceeded")
    finally:
        await ws_manager.disconnect(websocket)
```

> `fastapi-limiter` pour WebSocket, `slowapi` pour HTTP. Les deux utilisent Redis.

### Alternatives au rate limiting applicatif

- **Load balancer** (Nginx, HAProxy)
- **Reverse proxy**
- **API Gateway** (Kong, AWS API Gateway)

→ Rate limiting à l'infra si pas besoin de logique custom.

---

## 6. Throttling des streams

### Throttle au niveau du generator

```python
class AzureOpenAIChatClient:
    def __init__(self, throttle_rate=0.5):
        self.throttle_rate = throttle_rate

    async def chat_stream(self, prompt, mode="sse") -> AsyncGenerator[str, None]:
        stream = ...  # OpenAI stream
        async for chunk in stream:
            await asyncio.sleep(self.throttle_rate)  # ralentir sans bloquer l'event loop
            if chunk.choices[0].delta.content is not None:
                yield f"data: {chunk.choices[0].delta.content}\n\n" if mode == "sse" \
                      else chunk.choices[0].delta.content
```

### Traffic Shaping (niveau infra)

Utiliser `tc` (Linux) pour configurer des règles réseau sur les interfaces/containers Docker :
- Bandwidth limits
- Latency delays intentionnels
- Packet loss
- IP limits

> Complexe mais utile pour les services nécessitant un contrôle réseau fin (chat, video streaming).

---

## Récap — Couches de protection

```
[Internet]
    ↓
[Traffic Shaping / Load Balancer]     ← rate limiting infra
    ↓
[FastAPI Middleware]                   ← slowapi rate limiting (IP + user)
    ↓
[Input Guardrails]                    ← topical, prompt injection, moderation, PII
    ↓
[LLM Generation]                      ← throttled stream
    ↓
[Output Guardrails]                   ← hallucination, moderation, syntax
    ↓
[Response to Client]
```

## Dépendances

```bash
uv add slowapi fastapi-limiter coredis
# Redis pour le rate limiting multi-instances
docker run -d -p 6379:6379 redis
```

## Points clés à retenir

1. **Guardrails = probabilistes**. Combiner LLM auto-eval + règles classiques + ML pour réduire les faux positifs/négatifs
2. **Input guardrails en parallèle** avec la génération via `asyncio.create_task` + `asyncio.wait`
3. **Fast failure** : annuler la génération dès qu'un guardrail est triggered (`task.cancel()`)
4. **G-Eval** : framework d'évaluation par LLM (domain, criteria, steps, score 1-5) pour la modération output
5. **Rate limiting par user** (pas seulement IP) → plus robuste contre VPN/proxies
6. **Redis obligatoire** pour le rate limiting multi-instances en production
7. **Throttle les streams** avec `asyncio.sleep()` → ne bloque pas l'event loop
8. **Ne jamais donner de secrets au modèle** dans le system prompt (même avec guardrails)
9. **slowapi** pour HTTP, **fastapi-limiter** pour WebSocket
10. Pour les systèmes critiques → rate limiting à l'infra (load balancer, API gateway)