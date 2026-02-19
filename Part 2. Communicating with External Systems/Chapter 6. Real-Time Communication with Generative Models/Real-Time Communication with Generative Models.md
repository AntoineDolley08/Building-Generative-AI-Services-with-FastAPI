# Chapter 6 — Real-Time Communication with Generative Models

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Mécanismes de communication web](#1-mécanismes-de-communication-web)
2. [SSE — Server-Sent Events](#2-sse--server-sent-events)
3. [WebSocket](#3-websocket--communication-bidirectionnelle)
4. [Implémentation SSE](#4-implémentation-sse)
5. [Implémentation WebSocket](#5-implémentation-websocket)
6. [Gestion des exceptions WebSocket](#6-gestion-des-exceptions-websocket)
7. [Design d'APIs streaming](#7-design-dapis-streaming)

---

## 1. Mécanismes de communication web

### Le problème

HTTP classique = request-response unique. Le serveur répond **une seule fois** quand tout est prêt. Pour les modèles GenAI (génération lente, token par token), l'utilisateur attend longtemps puis reçoit un bloc de texte d'un coup.

**Solution** : streamer les données **au fur et à mesure** de la génération → meilleure UX, engagement maintenu.

### Quand NE PAS implémenter le streaming

- Le modèle ou l'API ne supporte pas le streaming
- La complexité ajoutée (server + client) n'est pas justifiée
- Gestion des déconnexions, reconnexions, perte de données
- Coût infra des connexions persistantes concurrentes
- Compatibilité navigateur avec le protocole choisi

### 5 mécanismes comparés

| Mécanisme | Direction | Connexion | Latence | Complexité | Use cases |
|-----------|-----------|-----------|---------|------------|-----------|
| **HTTP request-response** | Client → Server → Client | Fermée après réponse | Haute | Faible | REST APIs, services sans real-time |
| **Short polling** | Client → Server (répété) | Nouvelle à chaque intervalle | Moyenne | Faible | Dashboards analytics, status de jobs |
| **Long polling** | Client → Server (maintenu) | Ouverte jusqu'à réponse | Basse | Moyenne | Notifications, anciens chats |
| **SSE** | Server → Client (persistant) | Persistante, unidirectionnelle | Basse | Moyenne | Chat LLM, live feeds, dashboards temps réel |
| **WebSocket** | Bidirectionnelle | Persistante, full-duplex | Très basse | Haute | Chat multimédia, jeux, transcription live, collab |

---

## 2. SSE — Server-Sent Events

### Principe

Connexion HTTP **persistante et unidirectionnelle** (server → client). Le serveur pousse des événements au client tant que la connexion est ouverte.

```
Client ──GET (Accept: text/event-stream)──→ Server
Client ←──── data: token1 ────────────────── Server
Client ←──── data: token2 ────────────────── Server
Client ←──── data: [DONE] ───────────────── Server
                                              (connexion fermée)
```

### Handshake SSE

- Client : `Accept: text/event-stream`
- Server : `Content-Type: text/event-stream`, status 200
- Format des messages : `data: contenu\n\n`

### Avantages

- Simple à implémenter (HTTP standard)
- **Auto-reconnexion** intégrée via l'API `EventSource` du navigateur
- Event IDs pour reprendre les streams interrompus
- Pas de nouveau protocole (contrairement à WebSocket)
- **ChatGPT utilise SSE** pour streamer les réponses

### Limites

- **Unidirectionnel** : le client ne peut pas envoyer de données pendant le stream
- GET uniquement avec `EventSource` (workaround POST possible mais plus complexe)
- Texte uniquement (pas de binaire)

---

## 3. WebSocket — Communication bidirectionnelle

### Principe

Connexion **persistante et bidirectionnelle** (full-duplex) sur TCP. Après un handshake HTTP initial, le protocole bascule sur WebSocket (`ws://` ou `wss://`).

```
Client ──HTTP Upgrade──→ Server     (CONNECTING)
Client ←──── accept ────── Server     (OPEN)
Client ←→ messages ←→ Server          (échange libre)
Client ──── close ──────→ Server     (CLOSING)
Client ←──── close ────── Server     (CLOSED)
```

### Handshake HTTP Upgrade

```http
GET ws://localhost:8000/generate/text/stream HTTP/1.1
Connection: Upgrade
Upgrade: websocket
Sec-WebSocket-Key: 8WnhvZTK66EVvhDG++RD0w==
Sec-WebSocket-Version: 13
```

### Message frames

| Type | Contenu |
|------|---------|
| **Text frames** | Données UTF-8 |
| **Binary frames** | Données binaires |
| **Ping/Pong** | Contrôle de connexion (heartbeat) |
| **Close frame** | Fermeture gracieuse + status code |

### Lifecycle

```
CONNECTING → OPEN → CLOSING → CLOSED
```

### Avantages vs limites

| ✅ Avantages | ❌ Limites |
|---|---|
| Full-duplex (envoi + réception) | Plus complexe à implémenter |
| Latence très basse (TCP direct) | Serveur stateful → scaling plus difficile |
| Support binaire | Gestion manuelle de la reconnexion |
| Idéal pour multimédia/collab | Legacy browsers incompatibles |

### Webhook ≠ WebSocket

**Webhook** = server-to-server, event-driven, pas de connexion persistante. **WebSocket** = client-server, connexion persistante bidirectionnelle.

> En production, toujours utiliser `wss://` (WebSocket over TLS) comme on utilise `https://`.

---

## 4. Implémentation SSE

### 4a. Client streaming Azure OpenAI

```python
# stream.py
import asyncio
from typing import AsyncGenerator
from openai import AsyncAzureOpenAI

class AzureOpenAIChatClient:
    def __init__(self):
        self.aclient = AsyncAzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_endpoint=os.environ["OPENAI_API_ENDPOINT"],
            azure_deployment=os.environ["OPENAI_API_DEPLOYMENT"],
        )

    async def chat_stream(
        self, prompt: str, mode: str = "sse", model: str = "gpt-4o"
    ) -> AsyncGenerator[str, None]:
        stream = await self.aclient.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            stream=True,  # ← retourne un generator au lieu de la réponse complète
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield (
                    f"data: {chunk.choices[0].delta.content}\n\n"
                    if mode == "sse"
                    else chunk.choices[0].delta.content
                )
                await asyncio.sleep(0.05)  # throttle pour réduire le back pressure
        if mode == "sse":
            yield "data: [DONE]\n\n"

azure_chat_client = AzureOpenAIChatClient()
```

Points clés :
- `stream=True` → l'API retourne un async generator au lieu de la réponse complète
- Chaque chunk est préfixé `data: ` pour se conformer au spec SSE
- `asyncio.sleep(0.05)` → throttle pour ne pas submerger le client
- `[DONE]` → signal de fin de stream (convention OpenAI)
- Mode `sse` vs `ws` pour adapter le format de sortie

### 4b. Endpoint SSE (GET)

```python
# main.py
from fastapi.responses import StreamingResponse

@app.get("/generate/text/stream")
async def serve_stream_controller(prompt: str) -> StreamingResponse:
    return StreamingResponse(
        azure_chat_client.chat_stream(prompt),
        media_type="text/event-stream",
    )
```

Côté client avec `EventSource` (GET uniquement) :

```javascript
const source = new EventSource(
    'http://localhost:8000/generate/text/stream?prompt=' + encodeURIComponent(message)
);
source.addEventListener('open', () => console.log('Connected'));
source.addEventListener('message', (e) => {
    if (e.data === '[DONE]') { source.close(); return; }
    container.textContent += e.data;
});
source.addEventListener('error', (e) => { source.close(); });
```

### 4c. Endpoint SSE (POST)

Le `EventSource` ne supporte que GET. Pour POST → utiliser `fetch` manuellement :

```python
# main.py
@app.post("/generate/text/stream")
async def serve_stream_controller(prompt: Annotated[str, Body()]) -> StreamingResponse:
    return StreamingResponse(
        azure_chat_client.chat_stream(prompt),
        media_type="text/event-stream",
    )
```

```javascript
// Client — fetch + ReadableStream
async function stream(message) {
    const response = await fetch('/generate/text/stream', {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
        body: JSON.stringify({ prompt: message }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        container.textContent += decoder.decode(value);
    }
}
```

### 4d. Retry avec exponential backoff (client)

```javascript
async function stream(message, maxRetries = 3, initialDelay = 1000, backoffFactor = 2) {
    let delay = initialDelay;
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            // ... SSE connection logic
            return;
        } catch (error) {
            console.warn(`Attempt ${attempt + 1} failed`);
            if (attempt < maxRetries - 1) {
                await sleep(delay);
                delay *= backoffFactor;  // 1s → 2s → 4s
            } else {
                throw error;
            }
        }
    }
}
```

### 4e. Streaming depuis Hugging Face (vLLM)

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8080:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```

```python
from huggingface_hub import AsyncInferenceClient

client = AsyncInferenceClient("http://localhost:8080")

async def chat_stream(prompt: str) -> AsyncGenerator[str, None]:
    stream = await client.text_generation(prompt, stream=True)
    async for token in stream:
        yield token
        await asyncio.sleep(0.05)
```

### 4f. CORS — autoriser les requêtes cross-origin

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ⚠️ dev only — restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Alternative : monter les fichiers HTML en static pour rester sur la même origin :

```python
from fastapi.staticfiles import StaticFiles
app.mount("/pages", StaticFiles(directory="pages"), name="pages")
# → http://localhost:8000/pages/client-sse.html
```

---

## 5. Implémentation WebSocket

### 5a. Connection Manager

```python
# stream.py
from fastapi.websockets import WebSocket

class WSConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        await websocket.close()

    @staticmethod
    async def receive(websocket: WebSocket) -> str:
        return await websocket.receive_text()

    @staticmethod
    async def send(message: str | bytes | list | dict, websocket: WebSocket) -> None:
        if isinstance(message, str):
            await websocket.send_text(message)
        elif isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_json(message)

    async def broadcast(self, message: str | bytes | list | dict) -> None:
        for connection in self.active_connections:
            await self.send(message, connection)

ws_manager = WSConnectionManager()
```

Responsabilités :
- `connect` : accepte la connexion, l'ajoute à la liste active
- `disconnect` : ferme et retire de la liste
- `send` : dispatch selon le type (text, bytes, json)
- `broadcast` : envoie à **tous** les clients connectés (group chat, notifications)

### 5b. Endpoint WebSocket

```python
# main.py
from fastapi.websockets import WebSocket, WebSocketDisconnect

@app.websocket("/generate/text/streams")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:  # boucle tant que la connexion est ouverte
            prompt = await ws_manager.receive(websocket)
            async for chunk in azure_chat_client.chat_stream(prompt, "ws"):
                await ws_manager.send(chunk, websocket)
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.send("An internal server error has occurred", websocket)
    finally:
        await ws_manager.disconnect(websocket)
```

Points clés :
- `while True` : la connexion reste ouverte pour plusieurs échanges
- `WebSocketDisconnect` : exception levée quand le client ferme la connexion
- `finally` : **toujours** nettoyer (disconnect) même en cas d'erreur
- `asyncio.sleep(0.05)` : réduit les race conditions côté client

### 5c. Client WebSocket avec retry

```javascript
let ws;
let retryCount = 0;
const maxRetries = 5;
let isError = false;

function connectWebSocket() {
    ws = new WebSocket("ws://localhost:8000/generate/text/streams");
    ws.onopen = () => { retryCount = 0; isError = false; };
    ws.onmessage = (event) => { container.textContent += event.data; };
    ws.onclose = async () => {
        if (isError && retryCount < maxRetries) {
            await sleep(Math.pow(2, retryCount) * 1000);  // exponential backoff
            retryCount++;
            connectWebSocket();
        }
    };
    ws.onerror = (error) => { isError = true; ws.close(); };
}

// Envoyer un message
streamButton.addEventListener('click', () => {
    if (prompt && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(prompt);
    }
});

// Fermer manuellement
closeButton.addEventListener('click', () => { isError = false; ws.close(); });

connectWebSocket();  // connexion initiale
```

---

## 6. Gestion des exceptions WebSocket

### Différences avec HTTP

| HTTP | WebSocket |
|------|-----------|
| Status codes (4xx, 5xx) | Pas de status codes HTTP |
| `HTTPException` | Messages d'erreur via `send()` |
| Connexion fermée après réponse | Connexion maintenue ouverte |
| Stateless | Stateful |

### Status codes de fermeture WebSocket (RFC 6455)

| Code | Description |
|------|-------------|
| 1000 | Fermeture normale |
| 1001 | Client a navigué ailleurs / serveur down |
| 1002 | Violation du protocole WS |
| 1003 | Type de données non supporté |
| 1007 | Encodage inconsistant (non-UTF-8) |
| 1008 | Violation de politique (raison masquée pour sécurité) |
| 1011 | Erreur interne du serveur |

### Pattern de gestion d'erreur

```python
try:
    while True:
        # ... logique de messaging
except WebSocketDisconnect:
    logger.info("Client disconnected")      # normal
except Exception as e:
    logger.error(f"Error: {e}")
    await ws_manager.send("Error message", websocket)  # notifier le client
finally:
    await ws_manager.disconnect(websocket)  # toujours nettoyer
```

---

## 7. Design d'APIs streaming

### ❌ Anti-pattern : multiples endpoints streaming

```
/stream/greeting    → connexion 1
/stream/question    → connexion 2
/stream/followup    → connexion 3
```

Le client doit naviguer entre les endpoints, gérer l'état, les reconnexions → complexe des deux côtés.

### ✅ Pattern recommandé : un seul endpoint, routing via body/headers

```
POST /generate/text/stream
Body: { "prompt": "...", "mode": "question|followup|greeting", "context": {...} }
```

Le backend gère le routing, le state, le switch de modèle/prompt en interne. Le client n'a qu'un seul point d'entrée.

**Avantages** :
- State management simplifié côté client
- Business logic centralisée côté backend
- Le backend a accès aux BDD, services, prompts custom
- Un seul endpoint pour switch de modèle, prompt, contexte

---

## SSE vs WebSocket — guide de choix

| Critère | SSE | WebSocket |
|---------|-----|-----------|
| **Direction** | Server → Client | Bidirectionnelle |
| **Protocole** | HTTP standard | TCP (après handshake HTTP) |
| **Reconnexion** | Automatique (EventSource) | Manuelle |
| **Format** | Texte uniquement | Texte + binaire |
| **Complexité** | Simple | Élevée |
| **Scaling** | Stateless-friendly | Stateful → plus difficile |
| **Use case GenAI** | Chat LLM (streaming réponses) | Multimédia, speech-to-text, collab |

### Règle de choix

> **SSE** par défaut pour le streaming LLM (c'est ce que ChatGPT utilise). **WebSocket** seulement si tu as besoin de communication bidirectionnelle dans la même connexion.

---

## Récap des dépendances

```bash
uv add openai aiohttp  # clients async
# Pas de dépendance supplémentaire pour SSE/WS — intégrés dans FastAPI via Starlette
```

## Points clés à retenir

1. **SSE = choix par défaut** pour le streaming LLM (simple, auto-reconnexion, HTTP natif)
2. **WebSocket** seulement pour le bidirectionnel (chat multimédia, transcription live)
3. `stream=True` sur les clients OpenAI/Anthropic → retourne un async generator
4. Toujours **throttler** le stream (`asyncio.sleep(0.05)`) pour ne pas submerger le client
5. **Exponential backoff** côté client pour la reconnexion
6. **Un seul endpoint streaming** avec routing via body/headers (pas N endpoints)
7. WebSocket = **stateful** → plus difficile à scaler, nécessite un connection manager
8. CORS middleware nécessaire si client et server sur des origins différentes
9. `wss://` en production (comme `https://`)
10. `finally: disconnect()` → **toujours nettoyer** les connexions WebSocket