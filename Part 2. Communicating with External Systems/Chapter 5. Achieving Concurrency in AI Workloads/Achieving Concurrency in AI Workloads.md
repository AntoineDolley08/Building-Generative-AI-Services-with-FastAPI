# Chapter 5 — Achieving Concurrency in AI Workloads

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Fondamentaux : concurrency vs parallelism](#1-fondamentaux--concurrency-vs-parallelism)
2. [Async programming avec FastAPI](#2-async-programming-avec-fastapi)
3. [Projet : Web Scraper async](#3-projet--web-scraper-async-talk-to-the-web)
4. [Projet : Module RAG](#4-projet--module-rag-talk-to-documents)
5. [Optimisation du model serving](#5-optimisation-du-model-serving)
6. [Gestion des tâches longues](#6-gestion-des-tâches-longues)
7. [Récap & cheat sheets](#7-récap--cheat-sheets)

---

## 1. Fondamentaux : concurrency vs parallelism

### Deux types d'opérations bloquantes

| Type | Cause | Exemples |
|------|-------|----------|
| **I/O-bound** | Attente d'une opération d'entrée/sortie | Requêtes réseau, appels API, lecture/écriture fichiers, requêtes BDD |
| **Compute-bound** | Calcul intensif CPU/GPU | Inférence AI, entraînement de modèles, rendering 3D, simulations |

### 3 stratégies de résolution

| Stratégie | Cible | Approche |
|-----------|-------|----------|
| **System optimization** | I/O-bound | Async programming, multithreading |
| **Model optimization** | Memory/compute-bound | Optimisations mémoire, serveurs d'inférence spécialisés |
| **Queuing system** | Tâches longues | Background tasks, files d'attente |

### Concurrency (1 core) vs Parallelism (multi-core)

**Concurrency** : gérer plusieurs tâches en **alternant** entre elles → illusion d'exécution simultanée.

```
Tâche A: ████░░░░████░░░░████
Tâche B:     ████░░░░████░░░░
             ↑ switch  ↑ switch
```

Implémentée via async IO (1 thread, event loop) ou multithreading (plusieurs threads, orchestré par l'OS).

**Parallelism** : exécuter plusieurs tâches **réellement en même temps** sur des cores séparés.

```
Core 1: ████████████████████  (Tâche A)
Core 2: ████████████████████  (Tâche B)
```

Implémenté via multiprocessing : chaque process a son propre core, sa mémoire, ses ressources isolées.

**Analogie restaurant** : concurrent = un seul employé alterne entre commandes et cuisine. Parallèle = plusieurs employés simultanément.

### Le GIL (Global Interpreter Lock)

Le GIL empêche plusieurs threads d'exécuter du code Python simultanément dans un même process.

- Un seul thread actif à la fois → le multithreading Python n'est pas parallèle
- Le CPU switch entre threads pendant les attentes I/O → utile pour l'I/O-bound
- Inutile pour le compute-bound (le process est occupé à calculer)
- Sera rendu optionnel dans de futures versions (PEP 703 — free-threaded Python)

### 3 modèles d'exécution Python

| Modèle | Cores/Threads | Bon pour |
|--------|--------------|----------|
| **Synchrone** | 1 core, 1 thread, séquentiel | Apps mono-utilisateur, prototypage |
| **Concurrent non-parallèle** | 1 core, event loop ou multi-threads | I/O-bound (réseau, BDD, fichiers) |
| **Concurrent et parallèle** | Multi-cores, multi-process | Compute-bound, distribution de charge |

### Multithreading vs Multiprocessing

| | Multithreading | Multiprocessing |
|---|---------------|----------------|
| Mémoire | Partagée entre threads | Isolée par process |
| Communication | Simple (mémoire partagée) | Complexe (IPC nécessaire) |
| Crash | Un thread peut affecter les autres | Process isolés |
| Bon pour | I/O-bound | Compute-bound |
| Risques | Deadlocks, race conditions | Coût mémoire élevé, sync complexe |

### Cas particulier : APIs tierces = I/O-bound

```
[FastAPI] --HTTP async--> [OpenAI API]  ← compute-bound géré par OpenAI
    ↑
    └── I/O-bound de notre côté → async suffit
```

Pour les gros modèles self-hosted → ne pas scale avec du multiprocessing FastAPI (chaque worker recharge le modèle). Utiliser des serveurs d'inférence spécialisés : vLLM, Ray Serve, NVIDIA Triton.

---

## 2. Async programming avec FastAPI

### Sync vs Async — la différence concrète

```python
# Synchrone : 15 secondes (5 × 3, séquentiel)
def task():
    time.sleep(5)
for _ in range(3):
    task()

# Asynchrone : 5 secondes (3 tâches en parallèle)
async def task():
    await asyncio.sleep(5)
await asyncio.gather(task(), task(), task())
```

### Les mots-clés async/await

| Mot-clé | Rôle |
|---------|------|
| `async def` | Déclare une **coroutine** (fonction pausable/résumable) |
| `await` | Signale un point d'attente I/O → l'event loop peut switcher |
| `asyncio.run()` | Lance l'event loop et exécute une coroutine |
| `asyncio.gather()` | Exécute plusieurs coroutines **concurremment** |

Règles critiques :
- `await` uniquement dans une fonction `async def`
- Appeler une coroutine sans `await` → retourne un objet coroutine, ne s'exécute pas
- Code bloquant sync dans `async def` → bloque l'event loop entier

### Event Loop — le cœur d'asyncio

Boucle infinie qui orchestre les tâches async : vérifie les tâches prêtes → exécute jusqu'au prochain `await` → pause et switch → reprend quand l'I/O est terminé.

**Coroutines** : fonctions spéciales qui cèdent le contrôle sans perdre leur état (comme des générateurs).

### FastAPI : Event Loop + Thread Pool

```
                    ┌─────────────────────────────┐
                    │     FastAPI (1 CPU core)     │
                    │                              │
Requête async ──────→  Event Loop (main thread)    │  ← plus rapide
                    │       ↑ await ↓              │
Requête sync ───────→  Thread Pool (workers)       │  ← plus lent (switch GIL)
                    └─────────────────────────────┘
```

| Handler déclaré | Exécuté sur | Bloque le serveur ? |
|----------------|-------------|-------------------|
| `async def` + `await` | Event loop | ❌ Non |
| `def` (sync) | Thread pool | ❌ Non (délégué aux threads) |
| `async def` + code sync ⚠️ | Event loop | ✅ **OUI — DANGER** |

### ⚠️ Erreur critique : bloquer l'event loop

```python
# ❌ BLOQUE LE SERVEUR — async def avec client sync
@app.get("/block")
async def block_server():
    completion = sync_client.chat.completions.create(...)  # BLOQUANT !

# ✅ OK — sync def, délégué au thread pool
@app.get("/slow")
def slow_generator():
    completion = sync_client.chat.completions.create(...)

# ✅ OPTIMAL — async def avec client async
@app.get("/fast")
async def fast_generator():
    completion = await async_client.chat.completions.create(...)
```

> **Règle d'or** : si ta fonction utilise `async def`, tout le code I/O à l'intérieur doit être async. Sinon, utilise `def` et laisse FastAPI déléguer au thread pool.

### OpenAI sync vs async

```python
from openai import OpenAI, AsyncOpenAI

sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.post("/async")
async def async_generate(prompt: str = Body(...)):
    completion = await async_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content
```

### Rate limiting

Requêtes concurrentes → risque de dépasser les rate limits. Solution : exponential backoff (lib `stamina`).

### Pièges courants de l'async

| Piège | Conséquence |
|-------|-------------|
| Code sync dans `async def` | Bloque l'event loop |
| Oublier `await` devant une coroutine | Ne s'exécute pas |
| `await` sur une fonction non-async | `TypeError` |
| Lib sync (ex: `requests`) dans du code async | Annule les bénéfices |
| Pas de limite sur les opérations concurrentes | Memory leak |
| Ressources non fermées (connexions, buffers) | Memory leak |

> Recommandation : écrire en sync d'abord, puis migrer en async une fois la logique validée.

---

## 3. Projet : Web Scraper async (Talk to the Web)

### Architecture

```
[User prompt avec URLs] → [Regex extraction] → [aiohttp fetch concurrent] → [BeautifulSoup parse] → [LLM prompt enrichi]
```

### Code complet

```python
# scraper.py
import asyncio
import re
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

def extract_urls(text: str) -> list[str]:
    url_pattern = r"(?P<url>https?:\/\/[^\s]+)"
    return re.findall(url_pattern, text)

def parse_inner_text(html_string: str) -> str:
    soup = BeautifulSoup(html_string, "lxml")
    if content := soup.find("div", id="bodyContent"):
        return content.get_text()
    logger.warning("Could not parse the HTML content")
    return ""

async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        html_string = await response.text()
        return parse_inner_text(html_string)

async def fetch_all(urls: list[str]) -> str:
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[fetch(session, url) for url in urls],
            return_exceptions=True
        )
        success_results = [r for r in results if isinstance(r, str)]
        if len(results) != len(success_results):
            logger.warning("Some URLs could not be fetched")
        return " ".join(success_results)
```

Points clés : `aiohttp` au lieu de `requests` (async natif), `asyncio.gather()` pour fetch en parallèle, `return_exceptions=True` pour ne pas interrompre les autres fetches.

### Injection comme dépendance FastAPI

```python
# dependencies.py
async def get_urls_content(body: TextModelRequest = Body(...)) -> str:
    urls = extract_urls(body.prompt)
    if urls:
        try:
            return await fetch_all(urls)
        except Exception as e:
            logger.warning(f"Failed to fetch URLs - Error: {e}")
    return ""

# main.py
@app.post("/generate/text")
async def serve_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
) -> TextModelResponse:
    prompt = body.prompt + " " + urls_content
    output = generate_text(models["text"], prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

Limites : pas de gestion des pages dynamiques (SPA), anti-scraping, HTML mal structuré.

---

## 4. Projet : Module RAG (Talk to Documents)

### Pourquoi RAG ?

Les LLMs hallucinent (réponses confiantes mais incorrectes). RAG injecte des **données factuelles** dans le prompt sans fine-tuning coûteux.

### Pipeline complet

```
1. EXTRACTION    : PDF → texte brut (.txt)
2. TRANSFORMATION : texte → nettoyage → chunks → embedding vectors
3. STOCKAGE      : vectors + metadata → Qdrant (vector DB)
4. RETRIEVAL     : query → embed → cosine similarity search → top N chunks
5. GENERATION    : prompt original + chunks récupérés → LLM → réponse
```

### Architecture

```
[Streamlit UI]
    ├── Upload PDF → POST /upload
    │       ↓
    │   [save_file] → filesystem
    │       ↓ (background tasks)
    │   [pdf_text_extractor] → .txt
    │       ↓
    │   [load → clean → embed → store] → Qdrant DB
    │
    └── Chat → POST /generate/text
            ↓
        [embed(query)] → [Qdrant search] → top 3 chunks
            ↓
        prompt + urls_content + rag_content → LLM → réponse
```

### Étape 1 : Upload async

```python
# upload.py
import aiofiles
from aiofiles.os import makedirs

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 MB

async def save_file(file: UploadFile) -> str:
    await makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    async with aiofiles.open(filepath, "wb") as f:
        while chunk := await file.read(DEFAULT_CHUNK_SIZE):
            await f.write(chunk)
    return filepath
```

### Étape 2 : Pipeline de transformation

```python
# rag/extractor.py — PDF → TXT (sync, ⚠️ jamais dans async def)
from pypdf import PdfReader

def pdf_text_extractor(filepath: str) -> None:
    content = ""
    pdf_reader = PdfReader(filepath, strict=True)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            content += f"{page_text}\n\n"
    with open(filepath.replace("pdf", "txt"), "w", encoding="utf-8") as file:
        file.write(content)
```

```python
# rag/transform.py
import aiofiles
from transformers import AutoModel

embedder = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
)

async def load(filepath: str, chunk_size: int = 512) -> AsyncGenerator[str, Any]:
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        while chunk := await f.read(chunk_size):
            yield chunk  # async generator

def clean(text: str) -> str:
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\. ,", "", t)
    return t.replace("..", ".").replace(". .", ".").strip()

def embed(text: str) -> list[float]:
    return embedder.encode(text).tolist()
```

### Étape 3 : Stockage Qdrant

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

Repository pattern avec client async :

```python
# rag/repository.py
class VectorRepository:
    def __init__(self, host="localhost", port=6333):
        self.db_client = AsyncQdrantClient(host=host, port=port)

    async def create_collection(self, collection_name, size): ...
    async def create(self, collection_name, embedding_vector, original_text, source): ...
    async def search(self, collection_name, query_vector, retrieval_limit, score_threshold): ...
```

Service qui orchestre le pipeline :

```python
# rag/service.py
class VectorService(VectorRepository):
    async def store_file_content_in_db(self, filepath, chunk_size=512,
                                        collection_name="knowledgebase", collection_size=768):
        await self.create_collection(collection_name, collection_size)
        async for chunk in load(filepath, chunk_size):
            embedding_vector = embed(clean(chunk))
            await self.create(collection_name, embedding_vector, chunk, os.path.basename(filepath))

vector_service = VectorService()
```

### Étape 4 : Background Tasks

```python
@app.post("/upload")
async def file_upload_controller(file: Annotated[UploadFile, File(...)],
                                  bg_text_processor: BackgroundTasks):
    filepath = await save_file(file)
    bg_text_processor.add_task(pdf_text_extractor, filepath)          # sync → thread pool
    bg_text_processor.add_task(vector_service.store_file_content_in_db,
                               filepath.replace("pdf", "txt"), 512, "knowledgebase", 768)
    return {"filename": file.filename, "message": "File uploaded successfully"}
```

Les background tasks s'exécutent dans l'ordre, après la réponse au client.

### Étape 5 : Retrieval + Augmentation

```python
# dependencies.py
async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    rag_content = await vector_service.search("knowledgebase", embed(body.prompt), 3, 0.7)
    return "\n".join([c.payload["original_text"] for c in rag_content])

# main.py
@app.post("/generate/text")
async def serve_text_controller(
    request: Request,
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
    rag_content: str = Depends(get_rag_content),
) -> TextModelResponse:
    prompt = body.prompt + " " + urls_content + rag_content
    output = generate_text(models["text"], prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

### Semantic search & context ranking

- **Cosine similarity** : 1=identique, 0=aucun rapport, -1=opposé
- Résultats triés par score décroissant (le plus pertinent en premier)
- Important car les LLMs sont plus sensibles au contenu en **début de prompt**

### Limites et pistes d'amélioration

| Limite | Piste |
|--------|-------|
| Splitting naïf | Découper par phrases/paragraphes |
| Hallucinations persistantes | Query transformation, prompt compression |
| Dépassement context window | Sliding window |
| Retrieval imprécis | MMR, reranking, RAG Fusion |

### Structure des fichiers RAG

```
rag/
├── __init__.py
├── extractor.py      # PDF → TXT
├── transform.py      # load (async), clean, embed
├── repository.py     # VectorRepository (CRUD Qdrant)
└── service.py        # VectorService (pipeline complet)
```

---

## 5. Optimisation du model serving

### Le bottleneck des gros modèles

```
Disque → RAM (I/O-bound) → GPU VRAM (memory-bound) → Inférence (compute-bound)
                                    ↑
                            Le vrai goulot d'étranglement
```

Pour les gros modèles, l'inférence est **memory-bound** : charger les paramètres en VRAM prend plus de temps que le calcul.

### Techniques d'optimisation

**Compression** :

| Technique | Principe |
|-----------|----------|
| **Quantization** | Réduire la précision (float32 → float16 → int8) |
| **Pruning** | Supprimer les paramètres les moins importants |
| **Distillation** | Petit modèle "student" imite un gros "teacher" |
| **Fine-tuning** | Spécialiser un petit modèle |

**Inférence (transformers)** :

| Technique | Principe |
|-----------|----------|
| **Fast attention** | Optimiser le calcul d'attention sur GPU |
| **KV caching** | Réutiliser les attention maps calculées |
| **Paged attention** | Découper le KV cache en pages |
| **Request batching** | Grouper les requêtes |

### vLLM — serveur d'inférence LLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --api-key "your_secret_token"
```

Lance un serveur FastAPI **compatible OpenAI API**.

Intégration avec ton FastAPI :

```python
# models.py — requête async vers vLLM
async def generate_text(prompt: str, temperature: float = 0.7) -> str:
    messages = [{"role": "system", "content": "You are an AI assistant"},
                {"role": "user", "content": prompt}]
    headers = {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY')}"}
    async with aiohttp.ClientSession() as session:
        response = await session.post("http://localhost:8000/v1/chat",
                                       json={"temperature": temperature, "messages": messages},
                                       headers=headers)
        predictions = await response.json()
    return predictions["choices"][0]["message"]["content"]

# main.py — plus besoin de lifespan !
app = FastAPI()

@app.post("/generate/text")
async def serve_text_controller(request: Request, body: TextModelRequest):
    output = await generate_text(body.prompt, body.temperature)
    return TextModelResponse(content=output, ip=request.client.host)
```

### Batching

**Static** : batch fixe → attend la requête la plus longue → GPU sous-utilisé.

**Continuous/Dynamic** (vLLM) : requêtes insérées en continu dès qu'une place se libère → GPU saturé.

> vLLM avec continuous batching + paged attention = **4× moins de latence, 23× plus de throughput**.

### Paged Attention

Le KV cache des attention maps est découpé en **pages** (comme la mémoire virtuelle de l'OS) :

1. Partitioning du cache en pages de taille fixe
2. Lookup table : blocs logiques → blocs physiques
3. Selective loading : uniquement les pages nécessaires
4. Attention computation sur les pages chargées

→ Élimine la fragmentation mémoire, allocation dynamique.

### Latency vs Throughput

| Métrique | Définition | Unité |
|----------|-----------|-------|
| **Latency** | Temps requête → première réponse | Secondes |
| **Throughput** | Requêtes traitées par unité de temps | Tokens/minute (TPM) |

---

## 6. Gestion des tâches longues

### Background Tasks pour le batch processing

```python
async def batch_generate_image(prompt: str, count: int) -> None:
    images = generate_images(prompt, count)
    for i, image in enumerate(images):
        async with aiofiles.open(f"output_{i}.png", mode="wb") as f:
            await f.write(image)

@app.get("/generate/image/background")
def serve_image_background(background_tasks: BackgroundTasks, prompt: str, count: int):
    background_tasks.add_task(batch_generate_image, prompt, count)
    return {"message": "Task is being processed in the background"}
```

Le client reçoit la réponse immédiatement, les résultats sont sauvegardés pour récupération ultérieure (polling).

### Limites des BackgroundTasks

| Point | Détail |
|-------|--------|
| Pas de vrai parallélisme | Même event loop |
| CPU-bound bloque l'event loop | Après la réponse, mais bloque quand même |
| Pas de retry/exception avancé | Pas adapté aux pipelines critiques |

### Pour la production

| Outil | Rôle |
|-------|------|
| **Celery** | Queue manager |
| **Redis** | Cache + message broker |
| **RabbitMQ** | Message broker robuste |
| **Ray Serve** | Orchestration distribuée GPU |

---

## 7. Récap & cheat sheets

### Stratégies par type de bottleneck

| Bottleneck | Stratégie | Outils |
|-----------|-----------|--------|
| **I/O-bound** | Async programming | `asyncio`, `aiohttp`, `aiofiles`, `AsyncQdrantClient` |
| **Memory-bound** | Externaliser le model serving | vLLM, BentoML, Ray Serve |
| **Compute-bound** | Optimisations modèle + batching | Quantization, continuous batching, paged attention |
| **Tâches longues** | Background tasks + polling | `BackgroundTasks`, Celery + Redis |

### Ce qu'on a construit

| Feature | Technique |
|---------|-----------|
| Web scraper async | `aiohttp` + `BeautifulSoup` + `Depends()` |
| Module RAG complet | `pypdf` + `aiofiles` + Jina Embeddings + Qdrant + `BackgroundTasks` |
| Intégration vLLM | Requêtes async vers serveur OpenAI-compatible |
| Batch image generation | `BackgroundTasks` + sauvegarde async |

### Dépendances complètes

```bash
# Core
uv add openai aiohttp aiofiles python-multipart loguru

# Web scraper
uv add beautifulsoup4 lxml

# RAG
uv add qdrant-client pypdf transformers

# Model serving (Linux + GPU)
uv add vllm
```