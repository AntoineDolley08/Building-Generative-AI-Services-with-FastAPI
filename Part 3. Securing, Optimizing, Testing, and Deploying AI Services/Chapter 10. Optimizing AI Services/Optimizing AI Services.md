# Chapter 10 — Optimizing AI Services

> Notes du livre *Building Generative AI Services with FastAPI*

---

## Table des matières

1. [Vue d'ensemble des optimisations](#1-vue-densemble-des-optimisations)
2. [Batch Processing](#2-batch-processing)
3. [Caching](#3-caching)
4. [Model Quantization](#4-model-quantization)
5. [Structured Outputs](#5-structured-outputs)
6. [Prompt Engineering](#6-prompt-engineering)
7. [Fine-Tuning](#7-fine-tuning)

---

## 1. Vue d'ensemble des optimisations

Deux axes d'optimisation :

| Axe | Objectif | Techniques |
|-----|----------|------------|
| **Performance** | Latence, throughput, coûts | Batch processing, caching, quantization |
| **Qualité** | Précision et fiabilité des outputs | Structured outputs, prompt engineering, fine-tuning |

---

## 2. Batch Processing

### Le problème

Envoyer une requête API par entrée = lent, coûteux, et risque de rate limiting.

### Solution 1 : Schema structuré multi-items

Modifier le schema Pydantic pour demander une liste de résultats par requête :

```python
from pydantic import BaseModel

class BatchDocumentClassification(BaseModel):
    class Category(BaseModel):
        document_id: str
        category: list[str]

    categories: list[Category]
```

> Passer d'une requête par document à une requête pour N documents → réduction massive du nombre d'appels API.

### Solution 2 : Batch API (OpenAI)

Les providers comme OpenAI fournissent des APIs dédiées au batch processing avec :
- **-50% de coûts** par rapport aux endpoints standard
- Rate limits plus élevés
- Temps de complétion garanti (24h)

#### Étape 1 : Créer le fichier JSONL

```python
# batch.py
import json
from uuid import UUID

def create_batch_file(
    entries: list[str],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    filepath: str = "batch.jsonl",
    max_tokens: int = 1024,
) -> None:
    with open(filepath, "w") as file:
        for entry in entries:
            request = {
                "custom_id": f"request-{UUID()}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": entry},
                    ],
                    "max_tokens": max_tokens,
                },
            }
            file.write(json.dumps(request) + "\n")
```

#### Étape 2 : Soumettre et récupérer les résultats

```python
from openai import AsyncOpenAI
from openai.types import Batch

client = AsyncOpenAI()

async def submit_batch_job(filepath: str) -> Batch:
    file_response = await client.files.create(
        file=open(filepath, "rb"), purpose="batch"
    )
    return await client.batches.create(
        input_file_id=file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

async def retrieve_batch_results(batch_id: str):
    batch = await client.batches.retrieve(batch_id)
    if batch.status == "completed" and batch.output_file_id is not None:
        return await client.files.content(batch.output_file_id)
```

> Idéal pour les jobs qui ne nécessitent pas de réponse immédiate : classification, traduction, parsing de documents en masse.

---

## 3. Caching

### Stratégies de caching pour GenAI

| Stratégie | Principe | Use case |
|-----------|----------|----------|
| **Keyword caching** | Match exact clé-valeur | Fonctions/endpoints avec inputs répétés identiques |
| **Semantic caching** | Match par similarité sémantique | Queries utilisateur variables mais similaires |
| **Context/prompt caching** | Réutilisation des attention states | Longs system prompts, conversations multi-turn |

> Toujours considérer la fréquence de refresh du cache selon la nature des données et le niveau de staleness acceptable.

### Keyword caching (fastapi-cache)

```bash
pip install "fastapi-cache2[redis]"
```

#### Configuration dans le lifespan

```python
# main.py
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

@asynccontextmanager
async def lifespan(_: FastAPI):
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield

app = FastAPI(lifespan=lifespan)
```

#### Décoration des fonctions/endpoints

```python
from fastapi_cache.decorator import cache

@cache()
async def classify_document(title: str) -> str:
    ...

@router.post("/text")
@cache(expire=60)  # invalidation après 60s
async def serve_text_to_text_controller():
    ...
```

> Le décorateur `@cache()` doit toujours être **le dernier** (le plus proche de la fonction).

#### Directives Cache-Control

| Directive | Effet |
|-----------|-------|
| `max-age` | Durée (en secondes) pendant laquelle la réponse est fraîche |
| `no-cache` | Force la revalidation auprès du serveur |
| `no-store` | Interdit tout caching |
| `private` | Cache privé uniquement (navigateur) |

### Semantic caching

Le semantic caching retourne une réponse stockée basée sur des **inputs similaires** (pas identiques).

```
"How do you build generative services with FastAPI?"
≈ "What is the process of developing FastAPI services for GenAI?"
→ même réponse cachée retournée
```

> Réduit les appels API de 60-70% (cache hit rate) selon le use case et la taille de la base utilisateur.

#### Deux emplacements dans un système RAG

```
[Query] → [Semantic Cache] → hit? → réponse cachée
                            → miss? → [Vector Store] → documents → cache + LLM
```

- **Avant le LLM** → retourner une réponse cachée au lieu d'en générer une nouvelle
- **Avant le vector store** → enrichir le prompt avec des documents cachés

#### Implémentation from scratch avec Qdrant

```python
# cache_client.py
import uuid
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, PointStruct, ScoredPoint

class CacheClient:
    def __init__(self):
        self.db = AsyncQdrantClient(":memory:")
        self.cache_collection_name = "cache"

    async def initialize_database(self) -> None:
        await self.db.create_collection(
            collection_name=self.cache_collection_name,
            vectors_config=models.VectorParams(size=384, distance=Distance.EUCLID),
        )

    async def insert(self, query_vector: list[float], documents: list[str]) -> None:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=query_vector,
            payload={"documents": documents},
        )
        await self.db.upload_points(
            collection_name=self.cache_collection_name, points=[point]
        )

    async def search(self, query_vector: list[float]) -> list[ScoredPoint]:
        return await self.db.search(
            collection_name=self.cache_collection_name,
            query_vector=query_vector,
            limit=1,
        )
```

```python
# semantic_cache_service.py
import time
from loguru import logger
from transformers import AutoModel

class SemanticCacheService:
    def __init__(self, threshold: float = 0.35):
        self.embedder = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
        )
        self.euclidean_threshold = threshold
        self.cache_client = CacheClient()
        self.doc_db_client = DocumentStoreClient()

    def get_embedding(self, question) -> list[float]:
        return list(self.embedder.embed(question))[0]

    async def ask(self, query: str) -> str:
        vector = self.get_embedding(query)

        # 1. Chercher dans le cache
        if search_results := await self.cache_client.search(vector):
            for s in search_results:
                if s.score <= self.euclidean_threshold:
                    return s.payload["content"]  # cache hit

        # 2. Chercher dans le document store
        if db_results := await self.doc_db_client.search(vector):
            documents = [r.payload["content"] for r in db_results]
            await self.cache_client.insert(vector, documents)  # cacher pour la prochaine fois

        return "No answer available."
```

> Le **threshold** contrôle la sensibilité : trop haut → graph déconnecté (peu de hits), trop bas → faux positifs.

#### Alternative : GPT Cache

```bash
pip install gptcache
```

```python
# main.py
from gptcache import Config, cache
from gptcache.embedding import Onnx
from gptcache.processor.post import random_one
from gptcache.processor.pre import last_content
from gptcache.similarity_evaluation import OnnxModelEvaluation

@asynccontextmanager
async def lifespan(_: FastAPI):
    cache.init(
        post_func=random_one,               # sélection aléatoire parmi les résultats cachés
        pre_embedding_func=last_content,     # utiliser la dernière query comme clé
        embedding_func=Onnx().to_embeddings, # modèle ONNX pour les embeddings
        similarity_evaluation=OnnxModelEvaluation(),
        config=Config(similarity_threshold=0.75),
    )
    cache.set_openai_key()
    yield
```

> `gptcache` s'intègre automatiquement avec le client OpenAI — pas besoin de modifier le code d'appel existant.

#### Eviction policies

| Policy | Description | Use case |
|--------|-------------|----------|
| **FIFO** | Supprime les plus anciens | Items de même priorité |
| **LRU** | Supprime les moins récemment utilisés | Items récents plus susceptibles d'être réutilisés |
| **LFU** | Supprime les moins fréquemment utilisés | Items rares à supprimer en priorité |
| **MRU** | Supprime les plus récemment utilisés | Rarement utilisé |
| **RR** | Suppression aléatoire | Simple et rapide |

> Commencer par **LRU** avant de basculer vers d'autres policies.

### Context/prompt caching

Le context caching réutilise les **attention states pré-calculés** pour éviter de retraiter le même contexte à chaque requête.

#### Quand l'utiliser

- Chatbots avec de longs system prompts et conversations multi-turn
- Analyses répétées de fichiers vidéo/documents
- Requêtes récurrentes sur de grands corpus
- Analyse de repositories de code
- In-context learning avec beaucoup d'exemples

> Selon Anthropic : **-90% de coûts** et **-85% de latence** pour les longs prompts.

#### Avec l'API Anthropic

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    system=[
        {"type": "text", "text": "You are an AI assistant"},
        {
            "type": "text",
            "text": "<contenu d'un large document>",
            "cache_control": {"type": "ephemeral"},  # cache 5 min
        },
    ],
    messages=[{"role": "user", "content": "Summarize the documents..."}],
)
```

> `ephemeral` = cache de 5 minutes. Le context caching **introduit du state** entre les requêtes.

#### Avec l'API Google Gemini

```python
import datetime
import google.generativeai as genai
from google.generativeai import caching

corpus = genai.upload_file(path="corpus.txt")
cache = caching.CachedContent.create(
    model="models/gemini-1.5-flash-001",
    display_name="fastapi",       # identifiant du cache
    system_instruction="You are an expert AI engineer...",
    contents=[corpus],            # minimum 32,768 tokens
    ttl=datetime.timedelta(minutes=5),
)

model = genai.GenerativeModel.from_cached_content(cached_content=cache)
response = model.generate_content(["Introduce different characters..."])
```

> Le context caching **ne réduit pas les temps de réponse** de manière drastique mais **réduit significativement les coûts opérationnels**.

> Le caching de contexte ne cache pas les outputs — les réponses LLM restent non-déterministes.

---

## 4. Model Quantization

### Principe

La quantization projette les paramètres haute précision (FP32) vers des formats basse précision (FP16, INT8, INT4) via des facteurs de mise à l'échelle, réduisant la taille du modèle et accélérant l'inférence.

```
Model FP32 (haute précision) → Quantization → Model INT8 (basse précision)
                                                  ↓
                              Inférence rapide → Conversion → Output haute précision
```

### Formats de précision

| Format | Taille/param | Impact qualité | Use case |
|--------|-------------|---------------|----------|
| **FP32** | 4 bytes | Référence | Entraînement |
| **FP16** | 2 bytes | Minimal | Inférence standard |
| **BFLOAT16** | 2 bytes | Minimal (même range que FP32) | Bon compromis mémoire/qualité |
| **INT8** | 1 byte | Modéré | Inférence optimisée (le plus populaire) |
| **INT4** | 0.5 byte | Significatif | Devices low-power |
| **INT1** | 0.125 byte | Maximal | Recherche en cours |

### Bits d'un nombre flottant 32-bit

```
FP32  : [1 sign] [8 exponent] [23 mantissa]  → 32 bits
FP16  : [1 sign] [5 exponent] [10 mantissa]  → 16 bits
BF16  : [1 sign] [8 exponent] [7 mantissa]   → 16 bits (même range que FP32)
```

> Projeter FP32 → formats réduits = compresser la mantissa et ajuster l'exponent, avec une perte de précision minimale.

### Impact sur la taille des modèles Llama

| Modèle | Original | FP16 | 8 Bit | 4 Bit | 2 Bit |
|--------|----------|------|-------|-------|-------|
| **Llama 2 70B** | 140 GB | 128.5 GB | 73.23 GB | 36.20 GB | 28.59 GB |
| **Llama 3 8B** | 16.07 GB | 14.97 GB | 7.96 GB | 4.34 GB | 2.96 GB |

### Mémoire GPU requise

- **Inférence** : ~4 bytes/param en FP32 → un modèle 1B = 4 GB VRAM
- **Fine-tuning** : ~6x la mémoire d'inférence (gradients, optimizer states, activations) → un modèle 1B = 24 GB VRAM
- Prévoir **+5-8 GB de VRAM overhead** pour le chargement du modèle

### Quantization avec GPTQ

```bash
pip install auto-gptq optimum transformers accelerate
```

```python
import torch
from optimum.gptq import GPTQQuantizer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quantizer = GPTQQuantizer(
    bits=4,                                       # quantization en 4 bits
    dataset="c4",                                 # dataset de calibration
    block_name_to_quantize="model.decoder.layers", # quantize les decoder layers
    model_seqlen=2048,
)
quantized_model = quantizer.quantize_model(model, tokenizer)
```

> Un modèle 175B nécessite ~4 GPU hours sur NVIDIA A100 pour être quantizé. Vérifier d'abord si une version pré-quantizée existe sur Hugging Face.

---

## 5. Structured Outputs

### Le problème

Les LLMs ne garantissent pas un format de sortie — les systèmes downstream qui attendent du JSON peuvent crasher.

### Solution 1 : Schema Pydantic natif (OpenAI)

```python
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

client = AsyncOpenAI()

class DocumentClassification(BaseModel):
    category: str = Field(..., description="The category of the classification")

async def get_document_classification(title: str) -> DocumentClassification | str | None:
    response = await client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "classify the provided document into: ..."},
            {"role": "user", "content": title},
        ],
        response_format=DocumentClassification,  # schema Pydantic comme format
    )
    message = response.choices[0].message
    return message.parsed if message.parsed is not None else message.refusal
```

### Solution 2 : Prefill assistant (tout provider)

```python
import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

system_template = """
Classify the provided document into the following: ...
Provide responses in json: {"category": "string"}
"""

async def get_document_classification(title: str) -> dict:
    response = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1024,  # limiter les tokens → améliore robustesse et vitesse
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": title},
            {"role": "assistant", "content": "The document classification JSON is {"},
        ],
    )
    message = response.choices[0].message.content or ""
    try:
        return json.loads("{" + message[: message.rfind("}") + 1])
    except json.JSONDecodeError:
        return {"error": "Refusal response"}
```

> Le **prefill** de l'assistant (`"content": "{"`) force le modèle à compléter le JSON directement, sans preamble.

---

## 6. Prompt Engineering

### Template RCT (Role, Context, Task)

| Composant | Description | Exemple |
|-----------|-------------|---------|
| **Role** | Comment le modèle doit se comporter | "Tu es un expert en sécurité réseau" |
| **Context** | Scénario + informations de référence | Documents RAG, exemples, contraintes |
| **Task** | Instructions claires et non-ambiguës | "Classe ce document dans les catégories suivantes..." |

> Sans rôle explicite, le modèle utilise le contexte moyen de ses données d'entraînement. Un rôle précis améliore significativement la qualité.

### Techniques avancées de prompting

#### In-context learning

| Technique | Principe | Use case |
|-----------|----------|----------|
| **Zero-shot** | Pas d'exemples | Tâches simples (résumé, Q&A) |
| **Few-shot** | Quelques exemples fournis | Classification, extraction, sentiment analysis |
| **Dynamic few-shot** | Exemples injectés depuis un vector store | Réponses personnalisées, problèmes complexes |

> Les LLMs peuvent s'adapter à des exemples **sans modifier leurs poids** → c'est ce qui les distingue des modèles ML traditionnels.

#### Thought generation

| Technique | Prompt | Use case |
|-----------|--------|----------|
| **Zero-shot CoT** | "Let's think step by step..." | Raisonnement logique, maths |
| **Few-shot CoT** | "Let's think step by step..." + exemples | Q&A complexe, classification nuancée |
| **Thread of Thought (ThoT)** | "Walk me through in manageable parts..." | Dialogues, long-form content |

#### Decomposition

| Technique | Principe | Use case |
|-----------|----------|----------|
| **Least-to-most** | Décomposer en sous-problèmes sans les résoudre, puis résoudre un par un | Résolution de problèmes complexes |
| **Plan-and-solve** | Demander un plan puis demander de l'exécuter | Planification de projets |
| **Tree of Thoughts (ToT)** | Arbre de branches d'étapes à évaluer et résoudre | Exploration d'options multiples |

#### Autres techniques

| Technique | Principe |
|-----------|----------|
| **Ensembling** | Combiner plusieurs outputs de prompts différents (voting, agrégation) |
| **Self-criticism** | Le modèle critique et améliore sa propre réponse (Self-Refine, Chain of Verification) |
| **Agentic** | ReAct (Reason + Act), Toolformer → le modèle interagit avec des outils externes |

---

## 7. Fine-Tuning

### Quand fine-tuner ?

Le fine-tuning = **dernier recours** après avoir épuisé le prompt engineering et les structured outputs.

| Quand | Pas nécessaire |
|-------|---------------|
| Style/ton spécifique et constant | Le prompt engineering suffit |
| Domaine très spécialisé | Les structured outputs + few-shot suffisent |
| Réduction de la taille du prompt | Le context caching suffit |
| Haute fiabilité sur des tâches spécifiques | Le modèle standard avec CoT suffit |

### Workflow de fine-tuning (OpenAI)

```
1. Préparer le dataset (format JSONL)
2. Uploader le fichier
3. Lancer le fine-tuning job
4. Évaluer le modèle
5. Itérer si nécessaire
```

#### Format du dataset

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

#### Lancer le fine-tuning

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

# 1. Upload du dataset
file = await client.files.create(file=open("training.jsonl", "rb"), purpose="fine-tune")

# 2. Lancement du job
job = await client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini",
)

# 3. Récupérer le statut
status = await client.fine_tuning.jobs.retrieve(job.id)
print(status.fine_tuned_model)  # nom du modèle fine-tuné
```

> Le modèle fine-tuné est ensuite utilisable comme n'importe quel autre modèle via `model="ft:gpt-4o-mini:..."`.

---

## Récap

### Vue d'ensemble des techniques

```
Optimisations Performance          Optimisations Qualité
├── Batch Processing               ├── Structured Outputs
│   ├── Schema multi-items         │   ├── Schema Pydantic natif
│   └── Batch API (OpenAI)         │   └── Prefill assistant
├── Caching                        ├── Prompt Engineering
│   ├── Keyword (fastapi-cache)    │   ├── Template RCT
│   ├── Semantic (Qdrant/gptcache) │   ├── In-context learning
│   └── Context/prompt caching     │   ├── Thought generation (CoT)
└── Model Quantization             │   ├── Decomposition (ToT)
    ├── FP16, BF16, INT8, INT4     │   └── Ensembling, Self-criticism
    └── GPTQ (auto-gptq)          └── Fine-Tuning
                                       └── OpenAI fine-tuning API
```

## Dépendances

```bash
# Keyword caching
pip install "fastapi-cache2[redis]"

# Semantic caching
pip install gptcache qdrant-client transformers

# Context/prompt caching (Gemini)
pip install google-generativeai

# Model quantization
pip install auto-gptq optimum transformers accelerate

# Structured outputs
pip install openai pydantic
```

## Points clés à retenir

1. **Batch Processing** : utiliser des schemas multi-items ou la Batch API OpenAI (-50% de coûts, rate limits plus élevés)
2. **Keyword caching** (`fastapi-cache` + Redis) : simple et efficace pour les inputs répétés identiques
3. **Semantic caching** : réduit les appels API de 60-70% en matchant des queries similaires via embeddings
4. **Context/prompt caching** : réutilise les attention states → -90% de coûts et -85% de latence pour les longs prompts
5. **Eviction policies** : commencer par **LRU**, ajuster le similarity threshold par expérimentation
6. **Quantization** : INT8 est le format le plus populaire pour l'inférence. Chercher les modèles pré-quantizés sur Hugging Face avant de quantizer soi-même
7. **Structured outputs** : utiliser `response_format` avec un schema Pydantic quand le provider le supporte, sinon le **prefill assistant** comme fallback
8. **Prompt engineering** : template **RCT** (Role, Context, Task) comme minimum. Techniques avancées : CoT, ToT, few-shot dynamique
9. **Fine-tuning = dernier recours** : épuiser d'abord le prompt engineering, les structured outputs et le caching
10. **Ne jamais cacher les outputs LLM** quand des variations sont nécessaires — le context caching ne cache que les inputs, pas les réponses
